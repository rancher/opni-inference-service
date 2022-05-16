# Standard Library
import asyncio
import gc
import json
import logging
import os
import sys
import time
import zipfile
from collections import defaultdict
from io import StringIO

# Third Party
import numpy as np
import pandas as pd
from const import (
    ES_ENDPOINT,
    ES_PASSWORD,
    ES_USERNAME,
    IS_GPU_SERVICE,
    LOGGING_LEVEL,
    S3_BUCKET,
    SERVICE_TYPE,
    THRESHOLD,
)
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from nats.aio.errors import ErrTimeout
from opni_nats import NatsWrapper
from opnilog_predictor import OpniLogPredictor
from opnilog_trainer import consume_signal, train_model
from utils import load_cached_preds, s3_setup, save_cached_preds

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

nw = NatsWrapper()
es = AsyncElasticsearch(
    [ES_ENDPOINT],
    port=9200,
    http_compress=True,
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    use_ssl=True,
)
IS_CONTROL_PLANE_SERVICE = SERVICE_TYPE == "control-plane"
IS_RANCHER_SERVICE = SERVICE_TYPE == "rancher"

if SERVICE_TYPE == "control-plane" or SERVICE_TYPE == "rancher":
    script_source = 'ctx._source.anomaly_level = ctx._source.anomaly_predicted_count != 0 ? "Anomaly" : "Normal";'
else:
    script_source = 'ctx._source.anomaly_level = ctx._source.anomaly_predicted_count == 0 ? "Normal" : ctx._source.anomaly_predicted_count == 1 ? "Suspicious" : "Anomaly";'
script_source += "ctx._source.opnilog_confidence = params['opnilog_score'];"
script_for_anomaly = (
    "ctx._source.anomaly_predicted_count += 1; ctx._source.opnilog_anomaly = true;"
)


async def consume_logs(logs_queue):
    """
    coroutine to consume logs from NATS and put messages to the logs_queue
    """
    if IS_CONTROL_PLANE_SERVICE:
        await nw.subscribe(
            nats_subject="opnilog_cp_logs",
            payload_queue=logs_queue,
            nats_queue="workers",
        )
    elif IS_RANCHER_SERVICE:
        await nw.subscribe(
            nats_subject="opnilog_rancher_logs",
            payload_queue=logs_queue,
            nats_queue="workers",
        )
    else:
        await nw.subscribe(nats_subject="model_ready", payload_queue=logs_queue)
        if IS_GPU_SERVICE:
            await nw.subscribe(
                nats_subject="gpu_service_inference_internal", payload_queue=logs_queue
            )
        else:
            await nw.subscribe(
                nats_subject="preprocessed_logs",
                payload_queue=logs_queue,
                nats_queue="workers",
            )
            await nw.subscribe(
                nats_subject="gpu_service_predictions", payload_queue=logs_queue
            )


async def update_preds_to_es(df):
    """
    this function updates predictions and anomaly level to opensearch
    """

    async def doc_generator(df):
        for index, document in df.iterrows():
            doc_dict = document.to_dict()
            if "anomaly_level" in doc_dict:
                doc_dict["doc"] = dict()
                doc_dict["doc"]["anomaly_level"] = doc_dict["anomaly_level"]
                del doc_dict["anomaly_level"]
            yield doc_dict

    df["predictions"] = [1 if p < THRESHOLD else 0 for p in df["opnilog_confidence"]]
    df["_op_type"] = "update"
    df["_index"] = "logs"
    df.rename(columns={"log_id": "_id"}, inplace=True)
    if IS_CONTROL_PLANE_SERVICE or IS_RANCHER_SERVICE:
        df["anomaly_level"] = [
            "Anomaly" if p < THRESHOLD else "Normal" for p in df["opnilog_confidence"]
        ]
        try:
            await async_bulk(
                es, doc_generator(df[["_id", "_op_type", "_index", "anomaly_level"]])
            )
            logger.info(
                "Updated {} anomalies from {} logs to ES".format(
                    len(df[df["anomaly_level"] == "Anomaly"]),
                    len(df["anomaly_level"]),
                )
            )
        except Exception as e:
            logger.error(e)
    else:
        df["script"] = [
            {
                "source": (script_for_anomaly + script_source)
                if opnilog_score < THRESHOLD
                else script_source,
                "lang": "painless",
                "params": {"opnilog_score": opnilog_score},
            }
            for opnilog_score in df["opnilog_confidence"]
        ]
        try:
            await async_bulk(
                es, doc_generator(df[["_id", "_op_type", "_index", "script"]])
            )
            logger.info(
                "Updated {} anomalies from {} logs to ES".format(
                    len(df[df["predictions"] > 0]),
                    len(df["predictions"]),
                )
            )
        except Exception as e:
            logger.error(e)


async def infer_logs(logs_queue):
    """
    coroutine to get payload from logs_queue, call inference rest API and put predictions to elasticsearch.
    """
    s3_setup()
    saved_preds = defaultdict(float)
    load_cached_preds(saved_preds)
    opnilog_predictor = OpniLogPredictor()
    if IS_CONTROL_PLANE_SERVICE or IS_RANCHER_SERVICE:
        opnilog_predictor.load(save_path="model-output/")
    else:
        opnilog_predictor.download_from_s3()
        opnilog_predictor.load()

    max_payload_size = 128 if (IS_CONTROL_PLANE_SERVICE or IS_RANCHER_SERVICE) else 512
    last_time = time.time()
    pending_list = []
    while True:
        payload = await logs_queue.get()
        if payload is None:
            continue

        decoded_payload = json.loads(payload)
        if len(decoded_payload) == 1:
            pending_list.append(decoded_payload[0])
        else:
            df_payload = pd.read_json(
                StringIO(payload),
                dtype={"_id": object, "cluster_id": str, "ingest_at": str},
            )
            await run(df_payload, saved_preds, max_payload_size, opnilog_predictor)
            del df_payload
        start_time = time.time()
        if (start_time - last_time >= 1 and len(pending_list) > 0) or (
            len(pending_list) >= max_payload_size
        ):
            df_payload = pd.DataFrame(pending_list)
            last_time = start_time
            pending_list = []
            await run(df_payload, saved_preds, max_payload_size, opnilog_predictor)
            del df_payload
        del decoded_payload
        gc.collect()


async def run(df_payload, saved_preds, max_payload_size, opnilog_predictor):
    # Memorize predictions from GPU services.
    if "gpu_service_result" in df_payload.columns:
        logger.info("saved predictions from GPU service.")
        save_cached_preds(
            dict(zip(df_payload["masked_log"], df_payload["opnilog_confidence"])),
            saved_preds,
        )
    else:
        for i in range(0, len(df_payload), max_payload_size):
            df = df_payload[i : min(i + max_payload_size, len(df_payload))]

            is_log_cached = np.array([ml in saved_preds for ml in df["masked_log"]])
            df_cached_logs, df_new_logs = df[is_log_cached], df[~is_log_cached]

            if len(df_cached_logs) > 0:
                df_cached_logs["opnilog_confidence"] = [
                    saved_preds[ml] for ml in df_cached_logs["masked_log"]
                ]
                await update_preds_to_es(df_cached_logs)
                if IS_GPU_SERVICE:
                    logger.info("send cached results back.")
                    df_cached_logs["gpu_service_result"] = True
                    await nw.publish(
                        nats_subject="gpu_service_predictions",
                        payload_df=df_cached_logs.to_json().encode(),
                    )

            if len(df_new_logs) > 0:
                if not (
                    IS_GPU_SERVICE or IS_CONTROL_PLANE_SERVICE or IS_RANCHER_SERVICE
                ):
                    try:  # try to post request to GPU service. response would be b"YES" if accepted, b"NO" for declined/timeout
                        response = await nw.request(
                            "gpu_service_inference",
                            df_new_logs.to_json().encode(),
                            timeout=1,
                        )
                        response = response.data.decode()
                    except ErrTimeout:
                        logger.warning("request to GPU service timeout.")
                        response = "NO"
                    logger.info(f"{response} for GPU service")

                if (
                    IS_GPU_SERVICE
                    or IS_CONTROL_PLANE_SERVICE
                    or IS_RANCHER_SERVICE
                    or response == "NO"
                ):
                    unique_masked_logs = list(df_new_logs["masked_log"].unique())
                    logger.info(f" {len(unique_masked_logs)} unique logs to inference.")
                    pred_scores_dict = opnilog_predictor.predict(unique_masked_logs)

                    if pred_scores_dict is None:
                        logger.warning("fail to make predictions.")
                    else:
                        df_new_logs["opnilog_confidence"] = [
                            pred_scores_dict[ml] for ml in df_new_logs["masked_log"]
                        ]
                        save_cached_preds(pred_scores_dict, saved_preds)
                        await update_preds_to_es(df_new_logs)
                        if IS_GPU_SERVICE:
                            logger.info("send new results back.")
                            df_new_logs["gpu_service_result"] = True
                            await nw.publish(
                                nats_subject="gpu_service_predictions",
                                payload_df=df_new_logs.to_json().encode(),
                            )


async def init_nats():
    logger.info("Attempting to connect to NATS")
    await nw.connect()


async def get_pretrain_model():
    filenames = next(os.walk("/model/"), (None, None, []))[2]
    if len(filenames) == 1:
        model_zip_file = f"/model/{filenames[0]}"
        with zipfile.ZipFile(model_zip_file, "r") as zip_ref:
            zip_ref.extractall("./")
            logger.info("Extracted model from zipfile.")
        return True
    logger.error("did not find exactly 1 model")
    return False


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    logs_queue = asyncio.Queue(loop=loop)
    consumer_coroutine = consume_logs(logs_queue)
    inference_coroutine = infer_logs(logs_queue)

    task = loop.create_task(init_nats())
    loop.run_until_complete(task)

    if IS_CONTROL_PLANE_SERVICE or IS_RANCHER_SERVICE:
        init_model_task = loop.create_task(get_pretrain_model())
        model_loaded = loop.run_until_complete(init_model_task)
        if not model_loaded:
            sys.exit(1)

    if IS_CONTROL_PLANE_SERVICE or IS_RANCHER_SERVICE:
        loop.run_until_complete(
            asyncio.gather(
                inference_coroutine,
                consumer_coroutine,
            )
        )

    elif IS_GPU_SERVICE:
        job_queue = asyncio.Queue(loop=loop)
        signal_coroutine = consume_signal(job_queue, nw)
        training_coroutine = train_model(job_queue, nw)
        loop.run_until_complete(
            asyncio.gather(
                inference_coroutine,
                consumer_coroutine,
                signal_coroutine,
                training_coroutine,
            )
        )
    else:  # workload CPU SERVICE
        loop.run_until_complete(asyncio.gather(inference_coroutine, consumer_coroutine))

    try:
        loop.run_forever()
    finally:
        loop.close()
