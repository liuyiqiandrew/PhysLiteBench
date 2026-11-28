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
    metrics = {"diffusivity": float(m.diffusivity)}
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
    assert 0.72 <= errors["diffusivity"] <= 1.12 and errors["calibration"] < 0.000002


@pytest.mark.parametrize("group", ["first", "last", "resolved"])
def test_silent_flashes(errors, group):
    assert errors[group] < 0.004
