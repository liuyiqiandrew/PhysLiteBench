import json
from pathlib import Path

import numpy as np
import pytest

from model import Model


DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def metrics():
    calibration = json.loads((DATA / "calibration.json").read_text())
    truth = json.loads((DATA / "predictions.json").read_text())
    model = Model()
    assert model.fit(calibration) is model
    measured = [r["curvature"] for r in calibration]
    cal = np.asarray(model.predict_calibration(calibration))
    expected = np.array([r["value"] for r in truth])
    actual = np.asarray(model.predict([r["input"] for r in truth]))
    assert cal.shape == (len(calibration),) and actual.shape == expected.shape
    assert np.isfinite(cal).all() and np.isfinite(actual).all()
    groups = np.array([r["group"] for r in truth])
    result = {
        "width": float(model.width),
        "calibration_max_error": float(np.max(abs(cal - measured))),
        "max_errors": {g: float(np.max(abs(actual - expected)[groups == g]))
                       for g in sorted(set(groups))},
    }
    logs = Path("/logs/verifier")
    if logs.is_dir():
        (logs / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def test_fit(metrics):
    assert abs(metrics["width"] - 1.073) < 1e-5
    assert metrics["calibration_max_error"] < 1e-5


@pytest.mark.parametrize("group", ["moving_echo", "same_scan", "two_scans"])
def test_prediction(metrics, group):
    assert metrics["max_errors"][group] < 0.002
