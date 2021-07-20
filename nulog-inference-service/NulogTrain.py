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
from botocore.exceptions import EndpointConnectionError
from NuLogParser import LogParser
from opni_nats import NatsWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]


def train_nulog_model(minio_client, windows_folder_path):
    nr_epochs = 3
    num_samples = 0
    parser = LogParser()
    texts = parser.load_data(windows_folder_path)
    tokenized = parser.tokenize_data(texts, isTrain=True)
    parser.tokenizer.save_vocab()
    parser.train(tokenized, nr_epochs=nr_epochs, num_samples=num_samples)
    all_files = os.listdir("output/")
    if "nulog_model_latest.pt" in all_files and "vocab.txt" in all_files:
        logging.info("Completed training model")
        minio_client.meta.client.upload_file(
            "output/nulog_model_latest.pt", "nulog-models", "nulog_model_latest.pt"
        )
        minio_client.meta.client.upload_file(
            "output/vocab.txt", "nulog-models", "vocab.txt"
        )
        logging.info("Nulog model and vocab have been uploaded to Minio.")
    else:
        logging.info("Nulog model was not able to be trained and saved successfully.")
        return False
    return True


def minio_setup_and_download_data(minio_client):
    try:
        minio_client.meta.client.download_file(
            "training-logs", "windows.tar.gz", "windows.tar.gz"
        )
        logging.info("Downloaded logs from minio successfully")

        shutil.unpack_archive("windows.tar.gz", format="gztar")
    except EndpointConnectionError:
        logging.error(
            f"Could not connect to minio with endpoint_url={MINIO_ENDPOINT} aws_access_key_id={MINIO_ACCESS_KEY} aws_secret_access_key={MINIO_SECRET_KEY} "
        )
        logging.info("Job failed")
        return False

    try:
        minio_client.meta.client.head_bucket(Bucket="nulog-models")
        logging.info("nulog-models bucket exists")
    except botocore.exceptions.ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logging.info("nulog-models bucket does not exist so creating it now")
            minio_client.create_bucket(Bucket="nulog-models")
    return True


async def send_signal_to_nats(nw):
    await nw.publish(
        "gpu_trainingjob_status", b"JobEnd"
    )  ## tells the GPU service that a training job done.

    nulog_payload = {
        "bucket": "nulog-models",
        "bucket_files": {
            "model_file": "nulog_model_latest.pt",
            "vocab_file": "vocab.txt",
        },
    }
    await nw.publish(
        nats_subject="model_ready", payload_df=json.dumps(nulog_payload).encode()
    )
    logging.info(
        "Published to model_ready Nats subject that new Nulog model is ready to be used for inferencing."
    )


async def consume_signal(job_queue, nw):
    await nw.subscribe(
        nats_subject="gpu_service_training_internal", payload_queue=job_queue
    )


async def train_model(job_queue, nw):
    minio_client = boto3.resource(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )
    windows_folder_path = "windows/"
    while True:
        new_job = await job_queue.get()  ## TODO: should the metadata being used?

        res_download_data = minio_setup_and_download_data(minio_client)
        res_train_model = train_nulog_model(minio_client, windows_folder_path)
        if res_train_model:
            await send_signal_to_nats(nw)
        ## TODO: what to do if model training ever failed?


async def init_nats(nw):
    logging.info("Attempting to connect to NATS")
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
        logging.info(f"Nulog training failed. Exception {e}")
