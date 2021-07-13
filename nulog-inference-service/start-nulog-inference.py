# Standard Library
import asyncio
import gc
import json
import logging
import os
import time
import urllib.request
import zipfile
from collections import defaultdict

# Third Party
import pandas as pd
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from nats.aio.errors import ErrTimeout
from NulogServer import NulogServer
from opni_nats import NatsWrapper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
THRESHOLD = float(os.getenv("MODEL_THRESHOLD", 0.7))
ES_ENDPOINT = os.environ["ES_ENDPOINT"]
IS_CONTROL_PLANE_SERVICE = bool(os.getenv("IS_CONTROL_PLANE_SERVICE", False))
IS_GPU_SERVICE = bool(os.getenv("IS_GPU_SERVICE", False))


nw = NatsWrapper()
es = AsyncElasticsearch(
    [ES_ENDPOINT],
    port=9200,
    http_compress=True,
    http_auth=("admin", "admin"),
    verify_certs=False,
    use_ssl=True,
)

if IS_CONTROL_PLANE_SERVICE:
    script_source = 'ctx._source.anomaly_level = ctx._source.anomaly_predicted_count != 0 ? "Anomaly" : "Normal";'
else:
    script_source = 'ctx._source.anomaly_level = ctx._source.anomaly_predicted_count == 0 ? "Normal" : ctx._source.anomaly_predicted_count == 1 ? "Suspicious" : "Anomaly";'
script_source += "ctx._source.nulog_confidence = params['nulog_score'];"
script_for_anomaly = (
    "ctx._source.anomaly_predicted_count += 1; ctx._source.nulog_anomaly = true;"
)


async def consume_logs(logs_queue):
    """
    coroutine to consume logs from NATS and put messages to the logs_queue
    """
    if IS_CONTROL_PLANE_SERVICE:
        await nw.subscribe(
            nats_subject="preprocessed_logs_control_plane",
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
    # df["nulog_confidence"] = predictions
    df["predictions"] = [1 if p < THRESHOLD else 0 for p in df["nulog_confidence"]]
    # filter out df to only include abnormal predictions

    df["_op_type"] = "update"
    df["_index"] = "logs"
    df.rename(columns={"log_id": "_id"}, inplace=True)
    df["script"] = [
        {
            "source": (script_for_anomaly + script_source)
            if nulog_score < THRESHOLD
            else script_source,
            "lang": "painless",
            "params": {"nulog_score": nulog_score},
        }
        for nulog_score in df["nulog_confidence"]
    ]
    try:
        await async_bulk(es, doc_generator(df[["_id", "_op_type", "_index", "script"]]))
        logging.info(
            "Updated {} anomalies from {} logs to ES in {} seconds".format(
                len(df[df["predictions"] > 0]),
                len(masked_log),
                time.time() - start_time,
            )
        )
    except Exception as e:
        logging.error(e)


async def infer_logs(logs_queue):
    """
    coroutine to get payload from logs_queue, call inference rest API and put predictions to elasticsearch.
    """
    saved_preds = defaultdict(float)
    nulog_predictor = NulogServer()
    if IS_CONTROL_PLANE_SERVICE:
        nulog_predictor.load(save_path="control-plane-output/")
    else:
        nulog_predictor.download_from_minio()
        nulog_predictor.load()

    async def doc_generator(df):
        for index, document in df.iterrows():
            doc_dict = document.to_dict()
            yield doc_dict

    max_payload_size = 128 if IS_CONTROL_PLANE_SERVICE else 512
    while True:
        payload = await logs_queue.get()
        if payload is None:
            continue

        start_time = time.time()
        decoded_payload = json.loads(payload)
        ## TODO: testing decode with pd.read_json first to reduce unnecessary decode.
        ## logic: df = pd.read_json(payload, dtype={"_id": object})
        ## if "bucket" in df: reload nulog model
        if "bucket" in decoded_payload and decoded_payload["bucket"] == "nulog-models":
            if IS_CONTROL_PLANE_SERVICE:
                nulog_predictor.load(save_path="control-plane-output/")
            else:
                nulog_predictor.download_from_minio(decoded_payload)
                nulog_predictor.load()
            continue

        df_payload = pd.read_json(payload, dtype={"_id": object})
        if (
            "nulog_confidence" in df_payload.columns
        ):  ## memorize predictions from GPU services.
            for score, log in zip(
                df_payload["nulog_confidence"], df_payload["masked_log"]
            ):
                saved_preds[log] = score
            logging.info("saved predictions from GPU service.")
            continue

        for i in range(0, len(df_payload), max_payload_size):
            df = df_payload[i : min(i + max_payload_size, len(df_payload))]

            is_log_cached = df["masked_log"] in saved_preds
            df_cached_logs = df[is_log_cached]
            df_new_logs = df[~is_log_cached]
            df_cached_logs["nulog_confidence"] = [
                saved_preds[ml] for ml in df_cached_logs["masked_log"]
            ]
            await update_preds_to_es(df_cached_logs)

            if not (IS_GPU_SERVICE or IS_CONTROL_PLANE_SERVICE):
                try:  # try to post request to GPU service. response would be b"YES" if accepted, b"NO" for declined/timeout
                    response = await nw.nc.request(
                        "gpu_service_inference",
                        df_new_logs.to_json().encode(),
                        timeout=1,
                    )
                    response = response.decode()
                except ErrTimeout:
                    logging.warning("request to GPU service timeout.")
                    response = "NO"
                logging.info(f"{response} for GPU service")

            if response == "NO" or IS_GPU_SERVICE or IS_CONTROL_PLANE_SERVICE:
                unique_masked_logs = list(df_new_logs["masked_log"].unique())
                pred_scores_dict = nulog_predictor.predict(unique_masked_logs)

                if pred_scores is None:
                    logging.warning("fail to make predictions.")
                else:
                    df_new_logs["nulog_confidence"] = [
                        pred_scores_dict[ml] for ml in df_new_logs["masked_log"]
                    ]
                    for ml in pred_scores_dict:
                        saved_preds[ml] = pred_scores_dict[ml]
                    if IS_GPU_SERVICE:
                        nw.publish(
                            nats_subject="gpu_service_predictions",
                            payload=df_new_logs.to_json().encode(),
                        )
                    await update_preds_to_es(df_new_logs)

            del df
            del masked_log

        del decoded_payload
        del df_payload
        gc.collect()


async def init_nats():
    logging.info("Attempting to connect to NATS")
    await nw.connect()


async def get_pretrain_model():
    url = "https://opni-public.s3.us-east-2.amazonaws.com/pretrain-models/version.txt"
    try:
        latest_version = urllib.request.urlopen(url).read().decode("utf-8")
    except Exception as e:
        logging.error(e)
        logging.error("can't locate the version info from opni-public bucket")
        return False

    try:
        with open("version.txt") as fin:
            local_version = fin.read()
    except Exception as e:
        logging.warning(e)
        local_version = "None"
    logging.info(
        f"latest model version: {latest_version}; local model version: {local_version}"
    )

    if latest_version != local_version:
        urllib.request.urlretrieve(url, "version.txt")
        model_zip_file = f"control-plane-model-{latest_version}.zip"
        urllib.request.urlretrieve(
            f"https://opni-public.s3.us-east-2.amazonaws.com/pretrain-models/{model_zip_file}",
            model_zip_file,
        )
        with zipfile.ZipFile(model_zip_file, "r") as zip_ref:
            zip_ref.extractall("./")
        logging.info("update to latest model")
        return True
    else:
        logging.info("model already up to date")
        return False


async def schedule_update_pretrain_model(logs_queue):
    while True:
        await asyncio.sleep(86400)  # try to update after 24 hours
        update_status = await get_pretrain_model()
        if update_status:
            logs_queue.put(
                json.dumps({"bucket": "nulog-models"})
            )  # send a signal to reload model


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    logs_queue = asyncio.Queue(loop=loop)
    consumer_coroutine = consume_logs(logs_queue)
    inference_coroutine = infer_logs(logs_queue)

    task = loop.create_task(init_nats())
    loop.run_until_complete(task)

    if IS_CONTROL_PLANE_SERVICE:
        init_model_task = loop.create_task(get_pretrain_model())
        loop.run_until_complete(init_model_task)

    if IS_CONTROL_PLANE_SERVICE:
        loop.run_until_complete(
            asyncio.gather(
                inference_coroutine,
                consumer_coroutine,
                schedule_update_pretrain_model(logs_queue),
            )
        )
    else:
        loop.run_until_complete(asyncio.gather(inference_coroutine, consumer_coroutine))
    try:
        loop.run_forever()
    finally:
        loop.close()
