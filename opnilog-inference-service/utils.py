# Standard Library
import logging

# Third Party
import boto3
from botocore.config import Config
from const import (
    CACHED_PREDS_SAVEFILE,
    LOGGING_LEVEL,
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT,
    S3_SECRET_KEY,
    SAVE_FREQ,
)

s3_client = boto3.resource(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)


def s3_setup():
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


def save_cached_preds(new_preds: dict, saved_preds: dict):
    """
    save cached predictions of opnilog to s3 bucket
    """
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
    """
    reset all the cached preds if there's a model update.
    """
    bucket_name = S3_BUCKET
    saved_preds.clear()
    try:
        os.remove(CACHED_PREDS_SAVEFILE)
        s3_client.meta.client.delete_object(
            Bucket=bucket_name, Key=CACHED_PREDS_SAVEFILE
        )
    except Exception as e:
        logger.error("cached preds files failed to delete.")


def load_cached_preds(saved_preds: dict):
    """
    load cached preds from s3 bucket
    """
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
