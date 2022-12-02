# Standard Library
import asyncio
import json
import logging
import os
import shutil

# Third Party
import boto3
import botocore
from botocore.client import Config
from const import (
    DEFAULT_MODEL_NAME,
    ES_ENDPOINT,
    ES_PASSWORD,
    ES_USERNAME,
    LOGGING_LEVEL,
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT,
    S3_SECRET_KEY,
    TRAINING_DATA_PATH,
)
from elasticsearch import AsyncElasticsearch
from masker import LogMasker
from opni_nats import NatsWrapper
from opnilog_parser import LogParser

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)
masker = LogMasker()
ANOMALY_KEYWORDS = ["fail", "error", "fatal"]

es_instance = AsyncElasticsearch(
    [ES_ENDPOINT],
    port=9200,
    http_compress=True,
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    use_ssl=True,
)


async def get_all_training_data(payload):
    all_training_data = []
    scroll_id = ""
    query = payload["payload"]["query"]
    max_size = payload["payload"]["max_size"]
    first_iteration = True
    while True:
        if first_iteration:
            current_page = await es_instance.search(
                index="logs", body=query, scroll="1m", size=10000
            )
            first_iteration = False
        else:
            current_page = await es_instance.scroll(scroll_id=scroll_id, scroll="1m")
        results_hits = current_page["hits"]["hits"]
        if len(results_hits) > 0:
            scroll_id = current_page["_scroll_id"]
            for each_hit in results_hits:
                masked_log = masker.mask(each_hit["_source"]["log"])
                anomaly_keyword_matched = False
                for term in ANOMALY_KEYWORDS:
                    if term in masked_log:
                        anomaly_keyword_matched = True
                        break
                if not anomaly_keyword_matched:
                    all_training_data.append(masked_log)
                if len(all_training_data) == max_size:
                    return all_training_data
        else:
            return all_training_data


async def train_opnilog_model(nw, s3_client, query):
    """
    This function will be used to load the training data and then train the new OpniLog model.
    If during this process, there is any exception, it will return False indicating that a new OpniLog model failed to
    train. Otherwise, it will return True.
    """
    train_test_split = 0.9
    nr_epochs = 3
    num_samples = 0
    parser = LogParser()
    await nw.connect()
    model_training_payload = {"status": "training"}
    await nw.publish("model_update", json.dumps(model_training_payload).encode())
    # Load the training data.
    try:
        texts = await get_all_training_data(query)
        num_samples = min(len(texts), 128000)
    except Exception as e:
        logging.error(f"Unable to load data. {e}")
        return False
    try:
        if not os.path.exists("output/"):
            os.makedirs("output")
    except Exception as e:
        logging.error("Unable to create output folder.")
    # Check to see if the length of the training data is at least 1. Otherwise, return False.
    if len(texts) > 0:
        try:
            tokenized = parser.tokenize_data(texts, isTrain=True)
            parser.tokenizer.save_vocab()
            parser.train(tokenized, nr_epochs=nr_epochs, num_samples=num_samples)
            all_files = os.listdir("output/")
            if DEFAULT_MODEL_NAME in all_files and "vocab.txt" in all_files:
                logger.debug("Completed training model")
                s3_client.meta.client.upload_file(
                    "output/" + DEFAULT_MODEL_NAME, S3_BUCKET, DEFAULT_MODEL_NAME
                )
                s3_client.meta.client.upload_file(
                    "output/vocab.txt", S3_BUCKET, "vocab.txt"
                )
                logger.info("OpniLog model and vocab have been uploaded to S3.")
                shutil.rmtree("output/")
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


def s3_setup(s3_client):
    # Function to set up a S3 bucket if it does not already exist.
    try:
        s3_client.meta.client.head_bucket(Bucket=S3_BUCKET)
        logger.debug(f"{S3_BUCKET} bucket exists")
    except botocore.exceptions.ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logger.warning(f"{S3_BUCKET} bucket does not exist so creating it now")
            s3_client.create_bucket(Bucket=S3_BUCKET)
    return True


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


async def consume_signal(job_queue, nw):
    """
    This function subscribes to the Nats subject gpu_service_training_internal which will receive payload when it is
    time to train a new OpniLog model.
    """
    await nw.subscribe(
        nats_subject="gpu_service_training_internal", payload_queue=job_queue
    )


async def train_model(job_queue, nw):
    """
    This function will monitor the jobs_queue to see if any new training signal has been received. If it receives the
    signal, it will kick off a new training job and upon successful or failed training of a new OpniLog model, call
    the send_signal_to_nats method to send payload to the appropriate Nats subjects.
    """
    s3_client = boto3.resource(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )
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

    consumer_coroutine = consume_signal(job_queue, nw)
    training_coroutine = train_model(job_queue, nw)

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
