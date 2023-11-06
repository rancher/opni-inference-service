# Standard Library
import json
import logging

# Third Party
import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from const import (
    CACHED_PREDS_SAVEFILE,
    LOGGING_LEVEL,
    MODEL_STATS_ENDPOINT,
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT,
    S3_SECRET_KEY,
    SAVE_FREQ,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)


def put_model_stats(model_status, statistics):
    model_status = {"status": model_status, "statistics": statistics}
    try:
        result = requests.put(
            MODEL_STATS_ENDPOINT, data=json.dumps(model_status).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        logger.info(f"Put model training status : {result}")
    except Exception as e:
        logger.warning(f"Failed to post training status, error: {e}")


def get_s3_client():
    return boto3.resource(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def s3_setup(s3_client):
    # Function to set up a S3 bucket if it does not already exist.

    try:
        s3_client.meta.client.head_bucket(Bucket=S3_BUCKET)
        logger.debug(f"{S3_BUCKET} bucket exists")
    except ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logger.warning(f"{S3_BUCKET} bucket does not exist so creating it now")
            s3_client.create_bucket(Bucket=S3_BUCKET)
