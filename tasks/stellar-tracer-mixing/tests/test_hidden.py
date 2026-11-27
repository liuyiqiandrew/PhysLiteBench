import json
from pathlib import Path

import numpy as np
import pytest

from model import Model


DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def metrics():
    rows = json.loads((DATA / "calibration.json").read_text())
    hidden = json.loads((DATA / "predictions.json").read_text())
    model = Model()
    assert model.fit(rows) is model
    cal = np.array([r["value"] for r in rows])
    calibration_error = np.max(abs(model.predict([r["input"] for r in rows]) / cal - 1))
    errors = {}
    for group, cases in hidden.items():
        y = model.predict([r["input"] for r in cases])
        assert isinstance(y, np.ndarray) and y.shape == (len(cases),)
        errors[group] = float(np.max(abs(y / np.array([r["value"] for r in cases]) - 1)))
    result = {"exponent": float(model.exponent), "calibration": float(calibration_error),
              "max_relative_error": errors}
    logs = Path("/logs/verifier")
    if logs.is_dir():
        (logs / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def test_fit(metrics):
    assert abs(metrics["exponent"] - 2.73) < 0.01 and metrics["calibration"] < 0.002


@pytest.mark.parametrize("group", ["slow", "one_switch", "mixed_cycle", "two_switches"])
def test_prediction(metrics, group):
    assert metrics["max_relative_error"][group] < 0.02
