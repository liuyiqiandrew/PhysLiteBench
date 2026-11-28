import json
from pathlib import Path

import numpy as np
import pytest

from model import Model
from reference import hidden_inputs, predict

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "metadata.json").read_text())
LOG = Path("/logs/verifier/metrics.json")


@pytest.fixture(scope="module")
def metrics():
    records = json.loads((HERE / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(records) is model
    parameter = float(getattr(model, META["parameter"]))
    assert np.isfinite(parameter)
    true_parameter = META["true_parameter"]
    measured = np.array([r["value"] for r in records])
    sigma = np.array([r["sigma"] for r in records])
    calibration = model.predict([r["input"] for r in records])
    assert isinstance(calibration, np.ndarray) and calibration.shape == measured.shape
    assert np.isfinite(calibration).all()
    residual = (calibration - measured) / sigma
    results = {
        "parameters": {META["parameter"]: parameter},
        "parameter_relative_error_max": abs(parameter/true_parameter - 1),
        "calibration_chi2": float(residual @ residual / (len(records) - 1)),
        "hidden": {},
    }
    for name, cases in hidden_inputs().items():
        truth = predict(cases, true_parameter)
        actual = model.predict(cases)
        assert isinstance(actual, np.ndarray) and actual.shape == truth.shape
        assert np.isfinite(actual).all()
        scale = max(float(np.sqrt(np.mean(truth**2))), 0.02)
        results["hidden"][name] = float(np.sqrt(np.mean((actual - truth)**2)) / scale)
    if LOG.parent.exists():
        LOG.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


def test_private_calibration(metrics):
    assert metrics["calibration_chi2"] < 1.5


def test_parameter_recovery(metrics):
    assert metrics["parameter_relative_error_max"] < 0.03


@pytest.mark.parametrize("case", list(hidden_inputs()))
def test_prediction(metrics, case):
    assert metrics["hidden"][case] < META["prediction_limit"]
