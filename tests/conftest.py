# Standard Library
import json
import os

# Third Party
import pytest

os.environ["S3_ENDPOINT"] = "testing"
os.environ["S3_ACCESS_KEY"] = "testing"
os.environ["S3_SECRET_KEY"] = "testing"
os.environ["S3_BUCKET"] = "opni-drain-model"
os.environ["NATS_SERVER_URL"] = ""
os.environ["NATS_USERNAME"] = ""
os.environ["NATS_PASSWORD"] = ""
os.environ["ES_ENDPOINT"] = "testing_es"


@pytest.fixture
def test_data():
    test_text = []
    with open("tests/test-data/test.txt") as fin:
        for line in fin:
            test_text.append(json.loads(line)["log"])
    return test_text


TRAINING_DATA_PATH = "tests/test-data/training.txt"
