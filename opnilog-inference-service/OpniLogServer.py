# Standard Library
import logging
import os
from typing import List

# Third Party
import boto3
import inference as opniloginf
from botocore.config import Config
from OpniLogParser import using_GPU

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]
S3_ENDPOINT = os.environ["S3_ENDPOINT"]
S3_BUCKET = os.getenv("S3_BUCKET", "opni-nulog-models")
DEFAULT_MODELREADY_PAYLOAD = {
    "bucket": S3_BUCKET,
    "bucket_files": {
        "model_file": "nulog_model_latest.pt",
        "vocab_file": "vocab.txt",
    },
}


class OpniLogServer:
    def __init__(self, min_log_tokens):
        self.is_ready = False
        self.parser = None
        self.MIN_LOG_TOKENS = min_log_tokens

    def download_from_s3(
        self,
        decoded_payload: dict = DEFAULT_MODELREADY_PAYLOAD,
    ):
        s3_client = boto3.resource(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
        if not os.path.exists("output/"):
            os.makedirs("output")

        bucket_name = decoded_payload["bucket"]
        bucket_files = decoded_payload["bucket_files"]
        for k in bucket_files:
            try:
                s3_client.meta.client.download_file(
                    bucket_name, bucket_files[k], f"output/{bucket_files[k]}"
                )
            except Exception as e:
                logger.error(
                    "Cannot currently obtain necessary model files. Exiting function"
                )
                return

    def load(self, save_path="output/"):
        if using_GPU:
            logger.debug("inferencing with GPU.")
        else:
            logger.debug("inferencing without GPU.")
        try:
            self.parser = opniloginf.init_model(save_path=save_path)
            self.is_ready = True
            logger.info("OpniLog model gets loaded.")
        except Exception as e:
            logger.error(f"No OpniLog model currently {e}")

    def predict(self, logs: List[str]):
        """
        logs: masked logs
        """
        if not self.is_ready:
            logger.warning("Warning: OpniLog model is not ready yet!")
            return None

        # output = opniloginf.predict(self.parser, logs)
        output = []
        for log in logs:
            tokens = self.parser.tokenize_data([log], isTrain=False)
            if len(tokens[0]) < self.MIN_LOG_TOKENS:
                output.append(1)
            else:
                pred = (self.parser.predict(tokens))[0]
                output.append(pred)
        result_dict = {}
        for l, p in zip(logs, output):
            result_dict[l] = p
        return result_dict
