# Standard Library
import os

# Local
from opni_inference_service import inference, train
from tests.conftest import TRAINING_DATA_PATH


def test_train_load_text():
    data = train.load_text(TRAINING_DATA_PATH)

    assert len(data) > 0
    assert data == [
        "<klog_date> <num> <go_file_path> clientconn switching balancer to pick_first",
        "<klog_date> <num> <go_file_path> containermanager : discovered runtime cgroups name : <path>",
    ]


def test_train_opnilog_model(mocker):
    train.train_opnilog_model(TRAINING_DATA_PATH, duplicate=10)

    assert os.path.exists("output/opnilog_model_latest.pt")
    assert os.path.exists("output/vocab.txt")


def test_inference_main(test_data):
    test_parser = inference.init_model()

    preds = inference.predict(test_parser, test_data)
    test_threshold = 0.2  # this is bc in test env the training size too small

    assert preds[0] > test_threshold
    assert preds[1] > test_threshold
    assert preds[2] <= test_threshold
