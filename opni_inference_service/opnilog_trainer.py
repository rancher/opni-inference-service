# Standard Library
import asyncio
import copy
import gc
import json
import logging
import os
import random
import shutil
from collections import defaultdict

# Third Party
import numpy as np
from const import (
    DEFAULT_MODEL_NAME,
    DEFAULT_VOCAB_NAME,
    ES_ENDPOINT,
    ES_PASSWORD,
    ES_USERNAME,
    LOGGING_LEVEL,
    MAX_TRAINING_SAMPLE_SIZE,
    S3_BUCKET,
    TRAINING_DATA_PATH,
)
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from opni_nats import NatsWrapper
from utils import get_s3_client, s3_setup

# Local
from models.opnilog.masker import LogMasker
from models.opnilog.opnilog_parser import LogParser

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)
masker = LogMasker()
ANOMALY_KEYWORDS = ["fail", "error", "fatal"]

es_instance = Elasticsearch(
    [ES_ENDPOINT],
    port=9200,
    http_compress=True,
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    use_ssl=True,
)

RETRY_LIMIT = 5
MAX_FETCH_SIZE = 10000000
NUM_WORKER = 3


def get_all_training_data(payload):
    """
    get training data from Opensearch and mask them.
    """
    res_data = []
    query = payload["payload"]["query"]
    max_size = payload["payload"]["max_size"]
    num_logs_fetched = min(payload["payload"]["count"], max_size)
    num_retries = 0
    try:
        current_page = es_instance.search(
            index="logs", body=query, scroll="5m", size=10000
        )
    except Exception as e:
        logging.error("Unable to query Opensearch {e}")
        return res_data
    while len(current_hit := current_page["hits"]["hits"]) > 0:
        for h in current_hit:
            res_data.append(h["_source"]["log"])
            if len(res_data) == num_logs_fetched:
                return res_data
        while num_retries < RETRY_LIMIT:
            try:
                current_page = es_instance.scroll(
                    scroll_id=current_page["_scroll_id"], scroll="5m"
                )
                break
            except NotFoundError as e:
                num_retries += 1
    return res_data


def yield_all_training_data(payload):
    """
    yield training data from Opensearch
    """
    es_instance = Elasticsearch(
        [ES_ENDPOINT],
        port=9200,
        http_compress=True,
        http_auth=(ES_USERNAME, ES_PASSWORD),
        verify_certs=False,
        use_ssl=True,
    )

    query = payload["payload"]["query"]
    scroll_time = "5m"
    current_page = es_instance.search(
        index="logs", body=query, scroll=scroll_time, size=10000
    )
    logs_fetched = 0
    batch_data = []
    while len(current_hit := current_page["hits"]["hits"]) > 0:
        for h in current_hit:
            batch_data.append(h["_source"]["log"])
            logs_fetched += 1
        yield batch_data
        del batch_data
        gc.collect()
        batch_data = []
        current_page = es_instance.scroll(
            scroll_id=current_page["_scroll_id"], scroll=scroll_time
        )


def get_weights(data):
    """
    assign weights to masked dataset and make the distribution of different logs more balanced.
    """
    unique_sample_counter = defaultdict(int)
    for s in data:
        unique_sample_counter[str(s)] += 1

    for key in unique_sample_counter:
        count = unique_sample_counter[key]
        unique_sample_counter[key] = np.sqrt(count) / count  # the sqrt weight

    weights = np.array([unique_sample_counter[str(s)] for s in data])
    return weights


def preprocess_batch(payload):
    """
    the function that:
    1. masks data from Opensearch with Opni's custom masker
    2. applies weighted random shuffle
    3. yields each log.
    """
    downsample_ratio = payload["payload"]["downsample_ratio"]

    for batch in yield_all_training_data(payload):

        batch_masked = [masker.mask(b) for b in batch]
        # reduce batch_res accordingly using weighted random shuffle,
        weights = get_weights(batch_masked)
        if downsample_ratio < 1:
            size_reduce_to = int(downsample_ratio * len(batch))
            reduced_batch = random.choices(
                population=batch_masked, weights=weights, k=size_reduce_to
            )
        else:
            reduced_batch = batch_masked

        yield from reduced_batch
        del batch_masked
        del reduced_batch
        gc.collect()


def split_up_payload_query(payload, num_samples: int):
    """
    split payload query into NUM_WORKER queries
    """
    downsample_ratio = num_samples / payload["payload"]["count"]
    if NUM_WORKER == 1:
        payload["payload"]["downsample_ratio"] = downsample_ratio
        return [payload]
    res_payload = []

    start_ts = payload["payload"]["query"]["query"]["bool"]["filter"][0]["range"][
        "time"
    ]["gte"]
    end_ts = payload["payload"]["query"]["query"]["bool"]["filter"][0]["range"]["time"][
        "lte"
    ]
    ts_step = (end_ts - start_ts) // NUM_WORKER
    for i in range(NUM_WORKER):
        p = copy.deepcopy(payload)
        p["payload"]["query"]["query"]["bool"]["filter"][0]["range"]["time"] = {
            "gte": start_ts + i * ts_step,
            "lte": start_ts + (i + 1) * ts_step,
        }
        p["payload"]["downsample_ratio"] = downsample_ratio
        res_payload.append(p)
    return res_payload


async def train_opnilog_model(nw, s3_client, payload):
    """
    This function will be used to load the training data and then train the new OpniLog model.
    If during this process, there is any exception, it will return False indicating that a new OpniLog model failed to
    train. Otherwise, it will return True.
    """
    save_path = "output/"
    parser = LogParser(save_path=save_path)
    await nw.publish("model_update", json.dumps({"status": "training"}).encode())
    # Sleep for a few seconds so payload can be sent to the model_update Nats subject and it will be received before training is done.
    await asyncio.sleep(2)
    # Load the training data.
    training_method_threshold = 1000000
    log_count = payload["payload"]["count"]
    if log_count < training_method_threshold:
        # download all data if the dataset size is small, otherwise streaming.
        try:
            texts = get_all_training_data(payload)
            masked_logs = [masker.mask(log) for log in texts]
        except Exception as e:
            logging.error(f"Unable to load data. {e}")
            return False
        nr_epochs = 3
        # undersample if there are too many data, otherwise randomsample
        num_samples = (
            MAX_TRAINING_SAMPLE_SIZE if len(texts) > MAX_TRAINING_SAMPLE_SIZE else 0
        )
    else:
        nr_epochs = 2
        num_samples = (
            MAX_TRAINING_SAMPLE_SIZE * 3
        )  # 1 epoch so num_samples has to time 4
    # Check to see if the length of the training data is at least 1. Otherwise, return False.
    if log_count > 0:
        try:
            if log_count < training_method_threshold:
                tokenized = parser.tokenize_data(masked_logs, is_training=True)
                parser.tokenizer.save_vocab()
                parser.train(
                    tokenized,
                    nr_epochs=nr_epochs,
                    num_samples=num_samples,
                    put_results=True,
                )
            else:
                parser.train(
                    [],
                    nr_epochs=nr_epochs,
                    num_samples=num_samples,
                    put_results=True,
                    is_streaming=True,
                    iter_function=preprocess_batch,
                    iter_input_list=split_up_payload_query(payload, num_samples),
                )
                parser.tokenizer.save_vocab()
            all_files = os.listdir(save_path)
            if DEFAULT_MODEL_NAME in all_files and DEFAULT_VOCAB_NAME in all_files:
                logger.debug("Completed training model")
                s3_client.meta.client.upload_file(
                    os.path.join(save_path, DEFAULT_MODEL_NAME),
                    S3_BUCKET,
                    DEFAULT_MODEL_NAME,
                )
                s3_client.meta.client.upload_file(
                    os.path.join(save_path, DEFAULT_VOCAB_NAME),
                    S3_BUCKET,
                    DEFAULT_VOCAB_NAME,
                )
                logger.info("OpniLog model and vocab have been uploaded to S3.")
                shutil.rmtree(save_path)
                return True
            else:
                logger.error(
                    "OpniLog model was not able to be trained and saved successfully."
                )
                return False
        except Exception as e:
            logger.error(f"OpniLog model was not able to be trained. {e}")
            return False
    else:
        logger.error(
            "Cannot train OpniLog model as there was no training data present."
        )
        return False


async def send_signal_to_nats(nw, training_success):
    # Function that will send signal to Nats subjects gpu_trainingjob_status and model_update.
    await nw.connect()
    # Regardless of a successful training of OpniLog model, send JobEnd message to Nats subject gpu_trainingjob_status to make GPU available again.
    await nw.publish("gpu_trainingjob_status", b"JobEnd")

    # If OpniLog model has been successfully trained, send payload to model_update Nats subject that new model is ready to be uploaded from Minio.
    logger.info(f"training status : {training_success}")
    if training_success:
        opnilog_payload = {
            "bucket": S3_BUCKET,
            "bucket_files": {
                "model_file": DEFAULT_MODEL_NAME,
                "vocab_file": "vocab.txt",
            },
        }
        await nw.connect()
        await nw.publish(
            nats_subject="model_update", payload_df=json.dumps(opnilog_payload).encode()
        )
        logger.info(
            "Published to model_update Nats subject that new OpniLog model is ready to be used for inferencing."
        )


async def consume_signal_coroutine(job_queue, nw):
    """
    This function subscribes to the Nats subject gpu_service_training_internal which will receive payload when it is
    time to train a new OpniLog model.
    """
    await nw.subscribe(
        nats_subject="gpu_service_training_internal", payload_queue=job_queue
    )


async def train_model_coroutine(job_queue, nw):
    """
    This function will monitor the jobs_queue to see if any new training signal has been received. If it receives the
    signal, it will kick off a new training job and upon successful or failed training of a new OpniLog model, call
    the send_signal_to_nats method to send payload to the appropriate Nats subjects.
    """
    s3_client = get_s3_client()
    while True:
        new_job = await job_queue.get()  ## TODO: should the metadata being used?
        query = json.loads(new_job)
        logger.info("kick off a model training job...")
        res_s3_setup = s3_setup(s3_client)
        model_trained_success = await train_opnilog_model(nw, s3_client, query)
        await send_signal_to_nats(nw, model_trained_success)


async def init_nats(nw):
    logger.info("Attempting to connect to NATS")
    await nw.connect()


def main():
    loop = asyncio.get_event_loop()
    job_queue = asyncio.Queue(loop=loop)
    nw = NatsWrapper()

    consumer_coroutine = consume_signal_coroutine(job_queue, nw)
    training_coroutine = train_model_coroutine(job_queue, nw)

    task = loop.create_task(init_nats(nw))
    loop.run_until_complete(task)

    loop.run_until_complete(asyncio.gather(training_coroutine, consumer_coroutine))
    try:
        loop.run_forever()
    finally:
        loop.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"OpniLog training failed. Exception {e}")
