import json
from pathlib import Path
import numpy as np
import pytest
from model import Model


@pytest.fixture(scope="module")
def metrics():
    data = Path(__file__).parent / "data"
    rows = json.loads((data / "calibration.json").read_text())
    groups = json.loads((data / "hidden.json").read_text())
    model = Model()
    assert model.fit(rows) is model
    errors = {}
    for family, group in [("calibration", rows), *groups.items()]:
        pred = model.predict([r["input"] for r in group])
        assert isinstance(pred, np.ndarray) and pred.shape == (len(group),)
        assert pred.dtype.kind in "iuf" and np.isfinite(pred).all()
        errors[family] = float(max(abs(pred - [r["value"] for r in group])))
    result = {"gain": float(model.gain), "max_absolute_error": errors}
    path = Path("/logs/verifier/metrics.json")
    if path.parent.is_dir():
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def test_fit(metrics):
    assert 0.65 <= metrics["gain"] <= 1.15
    assert metrics["max_absolute_error"]["calibration"] < 2e-6


@pytest.mark.parametrize("family", ["bulk", "field_power", "cross_bias", "circulation"])
def test_film(family, metrics):
    assert metrics["max_absolute_error"][family] < 0.002
