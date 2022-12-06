# Standard Library
import json
import os

# Local
from opni_inference_service import inference, train


def test_load_text():
    data = train.load_text("tests/test-data/training.txt")
    assert len(data) > 0
    assert data == [
        "<klog_date> <num> <go_file_path> clientconn switching balancer to pick_first",
        "<klog_date> <num> <go_file_path> containermanager : discovered runtime cgroups name : <path>",
    ]


def test_train_opnilog_model(mocker):
    train.train_opnilog_model("tests/test-data/training.txt", duplicate=10)
    assert os.path.exists("output/opnilog_model_latest.pt")
    assert os.path.exists("output/vocab.txt")


def test_inference_main():
    test_parser = inference.init_model()
    test_text = []
    with open("tests/test-data/test.txt") as fin:
        for line in fin:
            test_text.append(json.loads(line)["log"])
    preds = inference.predict(test_parser, test_text)
    test_threshold = 0.2  # this is bc in test env the training size too small
    assert preds[0] > test_threshold
    assert preds[1] > test_threshold
    assert preds[2] <= test_threshold
