import json
from pathlib import Path
import numpy as np
import pytest
from model import Model


@pytest.fixture(scope="module")
def result():
    data = Path(__file__).parent / "data"
    cal = json.loads((data / "calibration.json").read_text())
    hidden = json.loads((data / "hidden.json").read_text())
    model = Model()
    assert model.fit(cal) is model
    errors = {}
    for group, rows in [("calibration", cal), *hidden.items()]:
        y = model.predict([r["input"] for r in rows])
        assert (
            isinstance(y, np.ndarray)
            and y.shape == (len(rows),)
            and y.dtype.kind in "iuf"
            and np.all(np.isfinite(y))
        )
        errors[group] = float(np.max(abs(y - [r["value"] for r in rows])))
    report = dict(diffusivity=float(model.diffusivity), max_absolute_error=errors)
    path = Path("/logs/verifier/metrics.json")
    if path.parent.is_dir():
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_gain(result):
    assert 0.72 <= result["diffusivity"] <= 1.10
    assert result["max_absolute_error"]["calibration"] < 2e-6


@pytest.mark.parametrize(
    "family", ["selector", "sticking", "timer", "wall_heat", "joint"]
)
def test_measurement(result, family):
    assert result["max_absolute_error"][family] < 0.004
