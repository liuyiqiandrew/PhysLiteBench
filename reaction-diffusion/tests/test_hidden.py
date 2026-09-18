import json
import os
from pathlib import Path
import pytest
from model import TransportModel
from reference import load_data, metrics

@pytest.fixture(scope="module")
def measured():
    data = load_data()
    result = metrics(TransportModel().fit(data), data)
    output = Path(os.environ.get("REACTION_METRICS_PATH", "/logs/verifier/metrics.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result

def test_calibration(measured):
    assert measured["calibration_reduced_chi2"] < 1.5

def test_parameter(measured):
    assert measured["relative_diffusivity_error"] < .05

@pytest.mark.parametrize("index", range(3))
def test_prediction(measured, index):
    assert measured["hidden_nrmse"][index] < .05
