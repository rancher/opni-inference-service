# Standard Library
import asyncio
import gc
import json
import logging
import os
import time
import zipfile

# Third Party
import pandas as pd
from const import LOGGING_LEVEL, S3_BUCKET, SERVICE_TYPE, THRESHOLD
from nats.aio.errors import ErrTimeout
from opni_nats import NatsWrapper
from opni_proto.log_anomaly_payload_pb import Payload, PayloadList
from opnilog_predictor import OpniLogPredictor
from opnilog_trainer import consume_signal, train_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

nw = NatsWrapper()
service_nats_subjects = {
    "control-plane": "opnilog_cp_logs",
    "rancher": "opnilog_rancher_logs",
    "longhorn": "opnilog_longhorn_logs",
}

IS_GPU_SERVICE = SERVICE_TYPE == "gpu"
IS_CPU_SERVICE = SERVICE_TYPE == "cpu"
IS_PRETRAINED_SERVICE = (
    SERVICE_TYPE == "control-plane"
    or SERVICE_TYPE == "rancher"
    or SERVICE_TYPE == "longhorn"
)
IS_WORKLOAD_SERVICE = IS_CPU_SERVICE or IS_GPU_SERVICE


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

    async def model_subscribe_handler(msg):
        model_data = msg.data
        await logs_queue.put(json.loads(model_data.decode()))

    async def gpu_subscribe_handler(msg):
        reply_subject = msg.reply
        await nw.publish(reply_subject, b"running")

    if IS_PRETRAINED_SERVICE:
        await nw.subscribe(
            nats_subject=service_nats_subjects[SERVICE_TYPE],
            payload_queue=logs_queue,
            nats_queue="workers",
            subscribe_handler=subscribe_handler,
        )
    else:
        await nw.subscribe(
            nats_subject="model_update",
            payload_queue=logs_queue,
            subscribe_handler=model_subscribe_handler,
        )
        if IS_GPU_SERVICE:
            await nw.subscribe(
                nats_subject="gpu_service_inference_internal",
                payload_queue=logs_queue,
                subscribe_handler=subscribe_handler,
            )
            await nw.subscribe(
                nats_subject="gpu_service_running",
                subscribe_handler=gpu_subscribe_handler,
            )
        else:
            await nw.subscribe(
                nats_subject="opnilog_workload_logs",
                payload_queue=logs_queue,
                nats_queue="workers",
                subscribe_handler=subscribe_handler,
            )


async def infer_logs(logs_queue):
    """
    coroutine to get payload from logs_queue, call inference rest API and put predictions to elasticsearch.
    """
    opnilog_predictor = OpniLogPredictor()
    if IS_PRETRAINED_SERVICE:
        opnilog_predictor.load(save_path="model-output/")
    else:
        opnilog_predictor.download_from_s3()
        opnilog_predictor.load()
    max_payload_size = 128 if (IS_PRETRAINED_SERVICE) else 512
    last_time = time.time()
    pending_list = []
    while True:
        payload = await logs_queue.get()
        if payload is None:
            continue
        if type(payload) is dict:
            if "bucket" in payload and payload["bucket"] == S3_BUCKET:
                opnilog_predictor.download_from_s3(payload)
                opnilog_predictor.load()
                continue
            if "status" in payload and (
                payload["status"] == "training" or payload["status"] == "reset"
            ):
                opnilog_predictor.reset_model()
                continue
        if not opnilog_predictor.is_ready:
            logging.info("No model currently trained.")
            continue

        if len(payload) == 1:
            pending_list.append(payload[0])
        else:
            df_payload = pd.DataFrame(payload)
            await run(df_payload, max_payload_size, opnilog_predictor)
            del df_payload
        start_time = time.time()
        if (start_time - last_time >= 1 and len(pending_list) > 0) or (
            len(pending_list) >= max_payload_size
        ):
            df_payload = pd.DataFrame(pending_list)
            last_time = start_time
            pending_list = []
            await run(df_payload, max_payload_size, opnilog_predictor)
            del df_payload
        del payload
        gc.collect()


async def run(df_payload, max_payload_size, opnilog_predictor):

    start_time = time.time()
    df_payload["inference_model"] = "opnilog"
    for i in range(0, len(df_payload), max_payload_size):
        df_batch = df_payload[i : min(i + max_payload_size, len(df_payload))]

        if len(df_batch) > 0:
            if IS_CPU_SERVICE:
                df_batch_list = list(map(lambda row: Payload(*row), df_batch.values))
                try:  # try to post request to GPU service. response would be b"YES" if accepted, b"NO" for declined/timeout
                    response = await nw.request(
                        "gpu_service_inference",
                        bytes(PayloadList(items=df_batch_list)),
                        timeout=1,
                    )
                    response = response.data.decode()
                except ErrTimeout:
                    logger.warning("request to GPU service timeout.")
                    response = "NO"
                except KeyError:
                    logger.warning("unable to send request to GPU service.")
                    response = "No"
                logger.info(f"{response} for GPU service")

            if not IS_CPU_SERVICE or response == "NO":
                unique_masked_logs = list(df_batch["masked_log"].unique())
                logger.info(f" {len(unique_masked_logs)} unique logs to inference.")
                pred_scores_dict = opnilog_predictor.predict(unique_masked_logs)

                if pred_scores_dict is None:
                    logger.warning("fail to make predictions.")
                else:
                    df_batch["opnilog_confidence"] = [
                        pred_scores_dict[ml] for ml in df_batch["masked_log"]
                    ]
                    df_batch["anomaly_level"] = [
                        "Anomaly" if p < THRESHOLD else "Normal"
                        for p in df_batch["opnilog_confidence"]
                    ]
                    df_batch_list = list(
                        map(lambda row: Payload(*row), df_batch.values)
                    )
                    if IS_WORKLOAD_SERVICE:
                        await nw.publish(
                            "model_inferenced_workload_logs",
                            bytes(PayloadList(items=df_batch_list)),
                        )
                    else:
                        await nw.publish(
                            "model_inferenced_pretrained_logs",
                            bytes(PayloadList(items=df_batch_list)),
                        )
    end_time = time.time()
    time_elapsed = end_time - start_time
    logging.info(f"Time elapsed here for model inferencing is {time_elapsed} seconds")


async def init_nats():
    logger.info("Attempting to connect to NATS")
    await nw.connect()


def get_pretrain_model():
    filenames = next(os.walk("/model/"), (None, None, []))[2]
    if len(filenames) == 1:
        model_zip_file = f"/model/{filenames[0]}"
        with zipfile.ZipFile(model_zip_file, "r") as zip_ref:
            zip_ref.extractall("./")
            logger.info("Extracted model from zipfile.")
    else:
        logger.error("did not find exactly 1 model")
        raise Exception("Error getting pretrained model")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    logs_queue = asyncio.Queue(loop=loop)
    consumer_coroutine = consume_logs(logs_queue)
    inference_coroutine = infer_logs(logs_queue)

    task = loop.create_task(init_nats())
    loop.run_until_complete(task)

    if IS_PRETRAINED_SERVICE:
        get_pretrain_model()
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
