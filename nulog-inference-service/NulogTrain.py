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
from NuLogParser import LogParser
from opni_nats import NatsWrapper

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

S3_ENDPOINT = os.environ["S3_ENDPOINT"]
S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]
S3_BUCKET = os.getenv("S3_BUCKET", "opni-nulog-models")
TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "/var/opni-data")


def train_nulog_model(s3_client, windows_folder_path):
    """
    This function will be used to load the training data and then train the new Nulog model.
    If during this process, there is any exception, it will return False indicating that a new Nulog model failed to
    train. Otherwise, it will return True.
    """
    nr_epochs = 10
    num_samples = 0
    parser = LogParser()
    # Load the training data.
    try:
        train_texts, validation_texts = parser.load_data(windows_folder_path)
    except Exception as e:
        logging.error("Unable to load data.")
        return False
    # Check to see if the length of the training data is at least 1. Otherwise, return False.
    if len(train_texts) > 0 and len(validation_texts) > 0:
        try:
            train_tokenized = parser.tokenize_data(train_texts, isTrain=True)
            validation_tokenized = parser.tokenize_data(validation_texts)
            parser.tokenizer.save_vocab()
            parser.train(
                train_tokenized,
                validation_tokenized,
                nr_epochs=nr_epochs,
                num_samples=num_samples,
            )
            all_files = os.listdir("output/")
            if "nulog_model_latest.pt" in all_files and "vocab.txt" in all_files:
                logger.debug("Completed training model")
                s3_client.meta.client.upload_file(
                    "output/nulog_model_latest.pt", S3_BUCKET, "nulog_model_latest.pt"
                )
                s3_client.meta.client.upload_file(
                    "output/vocab.txt", S3_BUCKET, "vocab.txt"
                )
                logger.info("Nulog model and vocab have been uploaded to S3.")
                shutil.rmtree("output/")
                return True
            else:
                logger.error(
                    "Nulog model was not able to be trained and saved successfully."
                )
                return False
        except Exception as e:
            logger.error("Nulog model was not able to be trained.")
            return False
    else:
        logger.error("Cannot train Nulog model as there was no training data present.")
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
    # Function that will send signal to Nats subjects gpu_trainingjob_status and model_ready.
    await nw.connect()
    # Regardless of a successful training of Nulog model, send JobEnd message to Nats subject gpu_trainingjob_status to make GPU available again.
    await nw.publish("gpu_trainingjob_status", b"JobEnd")

    # If Nulog model has been successfully trained, send payload to model_ready Nats subject that new model is ready to be uploaded from Minio.
    if training_success:
        nulog_payload = {
            "bucket": S3_BUCKET,
            "bucket_files": {
                "model_file": "nulog_model_latest.pt",
                "vocab_file": "vocab.txt",
            },
        }
        await nw.connect()
        await nw.publish(
            nats_subject="model_ready", payload_df=json.dumps(nulog_payload).encode()
        )
        logger.info(
            "Published to model_ready Nats subject that new Nulog model is ready to be used for inferencing."
        )


async def consume_signal(job_queue, nw):
    """
    This function subscribes to the Nats subject gpu_service_training_internal which will receive payload when it is
    time to train a new Nulog model.
    """
    await nw.subscribe(
        nats_subject="gpu_service_training_internal", payload_queue=job_queue
    )


async def train_model(job_queue, nw):
    """
    This function will monitor the jobs_queue to see if any new training signal has been received. If it receives the
    signal, it will kick off a new training job and upon successful or failed training of a new Nulog model, call
    the send_signal_to_nats method to send payload to the appropriate Nats subjects.
    """
    s3_client = boto3.resource(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )
    windows_folder_path = os.path.join(TRAINING_DATA_PATH, "windows")
    while True:
        new_job = await job_queue.get()  ## TODO: should the metadata being used?

        res_s3_setup = s3_setup(s3_client)
        model_trained_success = train_nulog_model(s3_client, windows_folder_path)
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
        logger.error(f"Nulog training failed. Exception {e}")
