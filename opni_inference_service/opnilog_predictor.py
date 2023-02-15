# Standard Library
import logging
import os
from typing import List

# Third Party
import inference as opniloginf
from const import (
    DEFAULT_MODELREADY_PAYLOAD,
    LOGGING_LEVEL,
    MIN_LOG_TOKENS,
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT,
    S3_SECRET_KEY,
)
from models.opnilog.opnilog_parser import using_GPU
from utils import get_s3_client

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)


class OpniLogPredictor:
    def __init__(self):
        self.is_ready = False
        self.parser = None
        self.s3_client = get_s3_client()

    def download_from_s3(
        self,
        decoded_payload: dict = DEFAULT_MODELREADY_PAYLOAD,
    ):
        if not os.path.exists("output/"):
            os.makedirs("output")

        bucket_name = decoded_payload["bucket"]
        bucket_files = decoded_payload["bucket_files"]
        for k in bucket_files:
            try:
                self.s3_client.meta.client.download_file(
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

    def reset_model(self):
        self.parser = None
        self.is_ready = False
        try:
            bucket = self.s3_client.Bucket(S3_BUCKET)
            for obj in bucket.objects.all():
                self.s3_client.Object(bucket.name, obj.key).delete()
            logging.info("Removed Opnilog files from S3.")

        except Exception as e:
            logging.error(f"Unable to remove model file from S3. {e}")

    def predict(self, logs: List[str]):
        """
        this methed defines the activity of model prediction for opnilog service
        """
        if not self.is_ready:
            logger.warning("Warning: OpniLog model is not ready yet!")
            return None

        output = []
        for log in logs:
            tokens = self.parser.tokenize_data([log], is_training=False)
            if len(tokens[0]) < MIN_LOG_TOKENS:
                output.append(1)
            else:
                pred = (self.parser.predict(tokens))[0]
                output.append(pred)
        result_dict = {}
        for l, p in zip(logs, output):
            result_dict[l] = p
        return result_dict
