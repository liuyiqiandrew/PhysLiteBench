import json
import os
from pathlib import Path
import pytest
from model import QubitModel
from reference import metrics, CHI2_LIMIT, PARAMETER_LIMIT, PREDICTION_LIMIT

@pytest.fixture(scope="module")
def measured():
    runs = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    result = metrics(QubitModel().fit(runs), runs)
    Path(os.environ.get("QUBIT_METRICS_PATH", "/logs/verifier/metrics.json")).write_text(json.dumps(result, indent=2)+"\n")
    return result

def test_calibration(measured):
    assert measured["calibration_reduced_chi2"] < CHI2_LIMIT

def test_parameter(measured):
    assert measured["relative_parameter_error"] < PARAMETER_LIMIT

@pytest.mark.parametrize("index", range(2))
def test_prediction(measured, index):
    assert measured["hidden_probability_rmse"][index] < PREDICTION_LIMIT
