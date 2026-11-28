import json
from pathlib import Path

import numpy as np
import pytest

from model import Model

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def metrics():
    calibration = json.loads((DATA / "calibration.json").read_text())
    model = Model()
    assert model.fit(calibration) is model
    cal = model.predict([r["input"] for r in calibration])
    result = {"intensity": float(model.intensity),
              "calibration_max_error": float(max(abs(cal - [r["value"] for r in calibration])))}
    for group, records in json.loads((DATA / "predictions.json").read_text()).items():
        predicted = model.predict([r["input"] for r in records])
        assert isinstance(predicted, np.ndarray) and predicted.shape == (len(records),)
        assert np.isfinite(predicted).all()
        result[group] = float(np.max(abs(predicted - [r["value"] for r in records])))
    log = Path("/logs/verifier/metrics.json")
    if log.parent.is_dir():
        log.write_text(json.dumps(result, indent=2) + "\n")
    return result


def test_fit(metrics):
    assert .4 <= metrics["intensity"] <= 1.3
    assert metrics["calibration_max_error"] < 1e-4


@pytest.mark.parametrize("group", ["separated", "pickup", "directed"])
def test_rate(metrics, group):
    assert metrics[group] < .003
