import json
import os
from pathlib import Path
import pytest
from model import ThermalModel
from reference import load_runs, metrics


@pytest.fixture(scope="module")
def measured():
    runs = load_runs(Path(__file__).parent)
    result = metrics(ThermalModel().fit(runs), runs)
    path = Path(os.environ.get("THERMAL_METRICS_PATH", "/logs/verifier/metrics.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def test_calibration(measured):
    assert measured["calibration_reduced_chi2"] < 1.5


def test_parameter(measured):
    assert measured["relative_conductance_error"] < .05


@pytest.mark.parametrize("index", range(3))
def test_prediction(measured, index):
    assert measured["hidden_nrmse"][index] < .05
