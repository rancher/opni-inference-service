# Third Party
import pytest

# Local
from opni_inference_service.opnilog_predictor import OpniLogPredictor


@pytest.fixture
def predictor(mocker):
    mocker.patch(
        "opni_inference_service.opnilog_predictor.get_s3_client", return_value=None
    )
    return OpniLogPredictor()


def test_load(predictor):
    predictor.load("tests/test-data/output")
    assert predictor.parser is not None
    assert predictor.is_ready is True


def test_predict(predictor, test_data):
    # test model not ready
    r = predictor.predict(test_data)
    assert r is None

    # test model is ready
    predictor.load("tests/test-data/output")

    res = predictor.predict(test_data)
    assert isinstance(res, dict)

    test_threshold = 0.2  # this is bc in test env the training size too small

    assert res[test_data[0]] > test_threshold
    assert res[test_data[1]] > test_threshold
    assert res[test_data[2]] <= test_threshold
