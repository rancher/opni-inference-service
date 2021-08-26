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
import boto3
import numpy as np
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from nats.aio.errors import ErrTimeout
from NulogServer import NulogServer
from NulogTrain import consume_signal, train_model
from opni_nats import NatsWrapper

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

THRESHOLD = float(os.getenv("MODEL_THRESHOLD", 0.7))
ES_ENDPOINT = os.environ["ES_ENDPOINT"]
ES_USERNAME = os.getenv("ES_USERNAME", "admin")
ES_PASSWORD = os.getenv("ES_PASSWORD", "admin")
S3_ENDPOINT = os.environ["S3_ENDPOINT"]
S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]
S3_BUCKET = os.getenv("S3_BUCKET", "opni-nulog-models")
IS_CONTROL_PLANE_SERVICE = bool(os.getenv("IS_CONTROL_PLANE_SERVICE", False))
IS_GPU_SERVICE = bool(os.getenv("IS_GPU_SERVICE", False))
CACHED_PREDS_SAVEFILE = (
    "control-plane-preds.txt"
    if IS_CONTROL_PLANE_SERVICE
    else "gpu-preds.txt"
    if IS_GPU_SERVICE
    else "cpu-preds.txt"
)
SAVE_FREQ = 25

nw = NatsWrapper()
es = AsyncElasticsearch(
    [ES_ENDPOINT],
    port=9200,
    http_compress=True,
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    use_ssl=True,
)
s3_client = boto3.resource(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
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
    async def doc_generator(df):
        for index, document in df.iterrows():
            doc_dict = document.to_dict()
            yield doc_dict

    df["predictions"] = [1 if p < THRESHOLD else 0 for p in df["nulog_confidence"]]
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
        logger.info(
            "Updated {} anomalies from {} logs to ES".format(
                len(df[df["predictions"] > 0]),
                len(df["predictions"]),
            )
        )
    except Exception as e:
        logger.error(e)


def s3_setup(s3_client):
    # Function to set up a S3 bucket if it does not already exist.
    try:
        s3_client.meta.client.head_bucket(Bucket=S3_BUCKET)
        logger.debug("{S3_BUCKET} bucket exists")
    except ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logger.warning("{S3_BUCKET} bucket does not exist so creating it now")
            s3_client.create_bucket(Bucket=S3_BUCKET)
    return True


def load_cached_preds(saved_preds: dict):

    bucket_name = S3_BUCKET
    try:
        s3_client.meta.client.download_file(
            bucket_name, CACHED_PREDS_SAVEFILE, CACHED_PREDS_SAVEFILE
        )
        with open(CACHED_PREDS_SAVEFILE) as fin:
            for line in fin:
                ml, score = line.split("\t")
                saved_preds[ml] = float(score)
    except Exception as e:
        logger.error("cached preds files do not exist.")
    logger.debug(f"loaded from cached preds: {len(saved_preds)}")
    return saved_preds


def save_cached_preds(new_preds: dict, saved_preds: dict):
    update_to_s3 = False
    bucket_name = S3_BUCKET
    with open(CACHED_PREDS_SAVEFILE, "a") as fout:
        for ml in new_preds:
            logger.debug("ml :" + str(ml))
            saved_preds[ml] = new_preds[ml]
            fout.write(ml + "\t" + str(new_preds[ml]) + "\n")
            if len(saved_preds) % SAVE_FREQ == 0:
                update_to_s3 = True
    logger.debug(f"saved cached preds, current num of cache: {len(saved_preds)}")
    if update_to_s3:
        try:
            s3_client.meta.client.upload_file(
                CACHED_PREDS_SAVEFILE, bucket_name, CACHED_PREDS_SAVEFILE
            )
        except Exception as e:
            logger.error("Failed to update predictions to s3.")


def reset_cached_preds(saved_preds: dict):
    bucket_name = S3_BUCKET
    saved_preds.clear()
    try:
        os.remove(CACHED_PREDS_SAVEFILE)
        s3_client.meta.client.delete_object(
            Bucket=bucket_name, Key=CACHED_PREDS_SAVEFILE
        )
    except Exception as e:
        logger.error("cached preds files failed to delete.")


async def infer_logs(logs_queue):
    """
    coroutine to get payload from logs_queue, call inference rest API and put predictions to elasticsearch.
    """
    s3_setup(s3_client)
    saved_preds = defaultdict(float)
    load_cached_preds(saved_preds)
    nulog_predictor = NulogServer()
    if IS_CONTROL_PLANE_SERVICE:
        nulog_predictor.load(save_path="control-plane-output/")
    else:
        nulog_predictor.download_from_s3()
        nulog_predictor.load()

    max_payload_size = 128 if IS_CONTROL_PLANE_SERVICE else 512
    while True:
        payload = await logs_queue.get()
        if payload is None:
            continue

        start_time = time.time()
        decoded_payload = json.loads(payload)
        if "bucket" in decoded_payload and decoded_payload["bucket"] == S3_BUCKET:
            # signal to reload model
            if IS_CONTROL_PLANE_SERVICE:
                nulog_predictor.load(save_path="control-plane-output/")
            else:
                nulog_predictor.download_from_s3(decoded_payload)
                nulog_predictor.load()
            reset_cached_preds(saved_preds)
            continue

        df_payload = pd.read_json(payload, dtype={"_id": object})
        if (
            "gpu_service_result" in df_payload.columns
        ):  ## memorize predictions from GPU services.
            logger.debug("saved predictions from GPU service.")
            save_cached_preds(
                dict(zip(df_payload["masked_log"], df_payload["nulog_confidence"])),
                saved_preds,
            )

        else:
            for i in range(0, len(df_payload), max_payload_size):
                df = df_payload[i : min(i + max_payload_size, len(df_payload))]

                is_log_cached = np.array([ml in saved_preds for ml in df["masked_log"]])
                df_cached_logs, df_new_logs = df[is_log_cached], df[~is_log_cached]

                if len(df_cached_logs) > 0:
                    df_cached_logs["nulog_confidence"] = [
                        saved_preds[ml] for ml in df_cached_logs["masked_log"]
                    ]
                    await update_preds_to_es(df_cached_logs)
                    if IS_GPU_SERVICE:
                        logger.debug("send cached results back.")
                        df_cached_logs["gpu_service_result"] = True
                        await nw.publish(
                            nats_subject="gpu_service_predictions",
                            payload_df=df_cached_logs.to_json().encode(),
                        )

                if len(df_new_logs) > 0:
                    if not (IS_GPU_SERVICE or IS_CONTROL_PLANE_SERVICE):
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
                        logger.debug(f"{response} for GPU service")

                    if IS_GPU_SERVICE or IS_CONTROL_PLANE_SERVICE or response == "NO":
                        unique_masked_logs = list(df_new_logs["masked_log"].unique())
                        logger.debug(
                            f" {len(unique_masked_logs)} unique logs to inference."
                        )
                        pred_scores_dict = nulog_predictor.predict(unique_masked_logs)

                        if pred_scores_dict is None:
                            logger.warning("fail to make predictions.")
                        else:
                            df_new_logs["nulog_confidence"] = [
                                pred_scores_dict[ml] for ml in df_new_logs["masked_log"]
                            ]
                            save_cached_preds(pred_scores_dict, saved_preds)
                            await update_preds_to_es(df_new_logs)
                            if IS_GPU_SERVICE:
                                logger.debug("send new results back.")
                                df_new_logs["gpu_service_result"] = True
                                await nw.publish(
                                    nats_subject="gpu_service_predictions",
                                    payload_df=df_new_logs.to_json().encode(),
                                )
            logger.info(
                f"payload size :{len(df_payload)}. processed in {(time.time() - start_time)} second"
            )

        del decoded_payload
        del df_payload
        gc.collect()


async def init_nats():
    logger.info("Attempting to connect to NATS")
    await nw.connect()


async def get_pretrain_model():
    url = "https://opni-public.s3.us-east-2.amazonaws.com/pretrain-models/version.txt"
    try:
        latest_version = urllib.request.urlopen(url).read().decode("utf-8")
    except Exception as e:
        logger.error(e)
        logger.error("can't locate the version info from opni-public bucket")
        return False

    try:
        with open("version.txt") as fin:
            local_version = fin.read()
    except Exception as e:
        logger.warning(e)
        local_version = "None"
    logger.info(
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
        logger.info("update to latest model")
        return True
    else:
        logger.info("model already up to date")
        return False


async def schedule_update_pretrain_model(logs_queue):
    while True:
        await asyncio.sleep(86400)  # try to update after 24 hours
        update_status = await get_pretrain_model()
        if update_status:
            logs_queue.put(
                json.dumps({"bucket": S3_BUCKET})
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
    else:  # CPU SERVICE
        loop.run_until_complete(asyncio.gather(inference_coroutine, consumer_coroutine))

    try:
        loop.run_forever()
    finally:
        loop.close()
