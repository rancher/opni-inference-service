# Standard Library
import asyncio
import json
import logging
import os
import shutil

# Third Party
from const import (
    DEFAULT_MODEL_NAME,
    DEFAULT_VOCAB_NAME,
    ES_ENDPOINT,
    ES_PASSWORD,
    ES_USERNAME,
    LOGGING_LEVEL,
    S3_BUCKET,
    TRAINING_DATA_PATH,
)
from elasticsearch import AsyncElasticsearch
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
MAX_TRAINING_SAMPLE_SIZE = 64000

es_instance = AsyncElasticsearch(
    [ES_ENDPOINT],
    port=9200,
    http_compress=True,
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    use_ssl=True,
)

RETRY_LIMIT = 5


def mask_logs(all_training_logs):
    all_masked_logs = []
    for log in all_training_logs:
        all_masked_logs.append(masker.mask(log))
    return all_masked_logs


async def get_all_training_data(payload):
    """
    get training data from Opensearch and mask them.
    """
    res_data = []
    query = payload["payload"]["query"]
    max_size = payload["payload"]["max_size"]
    num_logs_fetched = min(
        (await es_instance.count(index="logs", body=query))["count"], max_size
    )
    current_page = await es_instance.search(
        index="logs", body=query, scroll="1m", size=10000
    )
    while len(current_hit := current_page["hits"]["hits"]) > 0:
        for h in current_hit:
            res_data.append(masker.mask(h["_source"]["log"]))
            if len(res_data) >= num_logs_fetched:
                return res_data
        current_page = await es_instance.scroll(
            scroll_id=current_page["_scroll_id"], scroll="1m"
        )
    return res_data


async def train_opnilog_model(nw, s3_client, query):
    """
    This function will be used to load the training data and then train the new OpniLog model.
    If during this process, there is any exception, it will return False indicating that a new OpniLog model failed to
    train. Otherwise, it will return True.
    """
    save_path = "output/"
    parser = LogParser(save_path=save_path)
    await nw.connect()
    await nw.publish("model_update", json.dumps({"status": "training"}).encode())
    # Load the training data.
    try:
        texts = await get_all_training_data(query)
        masked_logs = mask_logs(texts)
    except Exception as e:
        logging.error(f"Unable to load data. {e}")
        return False

    # try:
    #     if not os.path.exists("output/"):
    #         os.makedirs("output")
    # except Exception as e:
    #     logging.error("Unable to create output folder.")

    nr_epochs = 3
    # undersample if there are too many data, otherwise randomsample
    num_samples = (
        MAX_TRAINING_SAMPLE_SIZE if len(texts) > MAX_TRAINING_SAMPLE_SIZE else 0
    )
    # Check to see if the length of the training data is at least 1. Otherwise, return False.
    if len(masked_logs) > 0:
        try:
            tokenized = parser.tokenize_data(masked_logs, isTrain=True)
            parser.tokenizer.save_vocab()
            parser.train(
                tokenized,
                nr_epochs=nr_epochs,
                num_samples=num_samples,
                put_results=True,
            )
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
