"""
all const and hyperparameters for opnilog service and model are defined here.
"""

# Standard Library
import json
import os

# logging level
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")

# opensearch config values
ES_ENDPOINT = os.environ["ES_ENDPOINT"]
ES_USERNAME = os.getenv("ES_USERNAME", "admin")
ES_PASSWORD = os.getenv("ES_PASSWORD", "admin")

# s3 config values
S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]
S3_ENDPOINT = os.environ["S3_ENDPOINT"]
S3_BUCKET = os.getenv("S3_BUCKET", "opni-nulog-models")
MODEL_STATS_ENDPOINT = "http://opni-internal:11080/ModelTraining/model/statistics"

# config values and hyperparameters for opnilg
DEFAULT_MODEL_NAME = "nulog_model_latest.pt"
DEFAULT_MODELREADY_PAYLOAD = {
    "bucket": S3_BUCKET,
    "bucket_files": {
        "model_file": "nulog_model_latest.pt",
        "vocab_file": "vocab.txt",
    },
}
TRAINING_DATA_PATH = os.getenv(
    "TRAINING_DATA_PATH", "/var/opni-data"
)  # only used by training


class HyperParameters:
    def __init__(self):
        self._MODEL_THRESHOLD = float(0.7)
        self._MIN_LOG_TOKENS = int(1)
        self._SERVICE_TYPE = "control-plane"
        if not (os.path.exists("/etc/opni/hyperparameters.json")):
            return
        f = open("/etc/opni/hyperparameters.json")
        data = json.load(f)
        if data is None:
            return
        if "modelThreshold" in data:
            self._MODEL_THRESHOLD = float(data["modelThreshold"])
        if "minLogTokens" in data:
            self._MIN_LOG_TOKENS = int(data["minLogTokens"])
        if "serviceType" in data:
            self._SERVICE_TYPE = data["serviceType"].lower()
        f.close()

    @property
    def MODEL_THRESHOLD(self):
        return self._MODEL_THRESHOLD

    @property
    def MIN_LOG_TOKENS(self):
        return self._MIN_LOG_TOKENS

    @property
    def SERVICE_TYPE(self):
        return self._SERVICE_TYPE


params = HyperParameters()
THRESHOLD = float(os.getenv("MODEL_THRESHOLD", params.MODEL_THRESHOLD))
MIN_LOG_TOKENS = int(os.getenv("MIN_LOG_TOKENS", params.MIN_LOG_TOKENS))
SERVICE_TYPE = os.getenv("SERVICE_TYPE", params.SERVICE_TYPE)

## these 2 const are specifically for caching predictions
CACHED_PREDS_SAVEFILE = (
    "control-plane-preds.txt"
    if SERVICE_TYPE == "control-plane"
    else "rancher-preds.txt"
    if SERVICE_TYPE == "rancher"
    else "gpu-preds.txt"
    if SERVICE_TYPE == "gpu"
    else "cpu-preds.txt"
)
SAVE_FREQ = 25
