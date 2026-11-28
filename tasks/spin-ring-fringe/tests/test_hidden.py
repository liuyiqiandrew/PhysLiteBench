import json
from pathlib import Path

import numpy as np
import pytest

from model import Model

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def metrics():
    calibration = json.loads((HERE / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(calibration) is model
    values = model.predict([r["input"] for r in calibration])
    residuals = (values - [r["value"] for r in calibration]) / [r["sigma"] for r in calibration]
    result = {"visibility": float(model.visibility), "chi2": float(residuals @ residuals / (len(values) - 1))}
    for family, records in json.loads((HERE / "data/hidden.json").read_text()).items():
        predicted = model.predict([r["input"] for r in records])
        assert isinstance(predicted, np.ndarray) and predicted.shape == (len(records),)
        assert np.isfinite(predicted).all()
        result[family] = float(np.sqrt(np.mean((predicted - [r["value"] for r in records])**2)))
    log = Path("/logs/verifier/metrics.json")
    if log.parent.exists():
        log.write_text(json.dumps(result, indent=2) + "\n")
    return result


def test_calibration(metrics):
    assert metrics["chi2"] < 2.5


def test_visibility(metrics):
    assert abs(metrics["visibility"] - .817) < .005


@pytest.mark.parametrize("family", ["anisotropic", "frustrated", "commuting", "large_xy"])
def test_signal(metrics, family):
    assert metrics[family] < .008
