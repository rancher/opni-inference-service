# Standard Library
import asyncio
import gc
import logging
import os
import sys
import time
import zipfile
from collections import defaultdict

# Third Party
import numpy as np
import pandas as pd
from const import IS_GPU_SERVICE, LOGGING_LEVEL, SERVICE_TYPE, THRESHOLD
from nats.aio.errors import ErrTimeout
from opni_nats import NatsWrapper
from opni_proto.log_anomaly_payload_pb import Payload, PayloadList
from opnilog_predictor import OpniLogPredictor
from opnilog_trainer import consume_signal, train_model
from utils import load_cached_preds, s3_setup, save_cached_preds

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

nw = NatsWrapper()
IS_CONTROL_PLANE_SERVICE = SERVICE_TYPE == "control-plane"
IS_RANCHER_SERVICE = SERVICE_TYPE == "rancher"


async def consume_logs(logs_queue):
    """
    coroutine to consume logs from NATS and put messages to the logs_queue
    """
    # This function will subscribe to the Nats subjects preprocessed_logs_control_plane and anomalies.
    async def subscribe_handler(msg):
        payload_data = msg.data
        log_payload_list = PayloadList()
        logs = (log_payload_list.parse(payload_data)).items
        await logs_queue.put(logs)

    if IS_CONTROL_PLANE_SERVICE:
        await nw.subscribe(
            nats_subject="opnilog_cp_logs",
            payload_queue=logs_queue,
            nats_queue="workers",
            subscribe_handler=subscribe_handler,
        )
    elif IS_RANCHER_SERVICE:
        await nw.subscribe(
            nats_subject="opnilog_rancher_logs",
            payload_queue=logs_queue,
            nats_queue="workers",
            subscribe_handler=subscribe_handler,
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
        if len(payload) == 1:
            pending_list.append(payload[0])
        else:
            df_payload = pd.DataFrame(payload)
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
        del payload
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
        start_time = time.time()
        df_payload["inference_model"] = "opnilog"
        for i in range(0, len(df_payload), max_payload_size):
            df = df_payload[i : min(i + max_payload_size, len(df_payload))]

            is_log_cached = np.array([ml in saved_preds for ml in df["masked_log"]])
            df_cached_logs, df_new_logs = df[is_log_cached], df[~is_log_cached]

            if len(df_cached_logs) > 0:
                df_cached_logs["opnilog_confidence"] = [
                    saved_preds[ml] for ml in df_cached_logs["masked_log"]
                ]
                df_cached_logs["anomaly_level"] = [
                    "Anomaly" if p < THRESHOLD else "Normal"
                    for p in df_cached_logs["opnilog_confidence"]
                ]
                df_cached_list = list(
                    map(lambda row: Payload(*row), df_cached_logs.values)
                )
                await nw.publish(
                    "inferenced_logs", bytes(PayloadList(items=df_cached_list))
                )
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
                        df_new_logs["anomaly_level"] = [
                            "Anomaly" if p < THRESHOLD else "Normal"
                            for p in df_new_logs["opnilog_confidence"]
                        ]
                        save_cached_preds(pred_scores_dict, saved_preds)
                        df_new_logs_list = list(
                            map(lambda row: Payload(*row), df_new_logs.values)
                        )
                        await nw.publish(
                            "inferenced_logs",
                            bytes(PayloadList(items=df_new_logs_list)),
                        )
                        if IS_GPU_SERVICE:
                            logger.info("send new results back.")
                            df_new_logs["gpu_service_result"] = True
                            await nw.publish(
                                nats_subject="gpu_service_predictions",
                                payload_df=df_new_logs.to_json().encode(),
                            )
        end_time = time.time()
        time_elapsed = end_time - start_time
        logging.info(
            f"Time elapsed here for model inferencing is {time_elapsed} seconds"
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
