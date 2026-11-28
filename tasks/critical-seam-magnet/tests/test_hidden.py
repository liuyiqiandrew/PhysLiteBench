from pathlib import Path
import json
import numpy as np
import pytest
from model import Model

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def errors():
    rows = json.loads((DATA / "calibration.json").read_text())
    m = Model()
    assert m.fit(rows) is m
    metrics = {"gain": float(m.gain)}
    groups = json.loads((DATA / "predictions.json").read_text())
    for name, rs in [("calibration", rows), *groups.items()]:
        values = m.predict([r["input"] for r in rs])
        assert isinstance(values, np.ndarray) and values.dtype.kind in "ifu"
        assert values.shape == (len(rs),) and np.isfinite(values).all()
        metrics[name] = float(max(abs(values - [r["value"] for r in rs])))
    path = Path("/logs/verifier/metrics.json")
    if path.parent.is_dir():
        path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def test_calibration(errors):
    assert 0.75 <= errors["gain"] <= 1.15 and errors["calibration"] < 0.00002


@pytest.mark.parametrize("group", ["overlap", "connected", "response"])
def test_spin_contrast(errors, group):
    assert errors[group] < 0.004
