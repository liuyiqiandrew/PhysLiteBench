import json
from pathlib import Path
import numpy as np
import pytest
from model import Model


DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def errors():
    calibration = json.loads((DATA / "calibration.json").read_text())
    groups = json.loads((DATA / "predictions.json").read_text())
    model = Model()
    assert model.fit(calibration) is model
    metrics = {"rate": float(model.rate), "max_absolute_error": {}}
    for name, rows in [("calibration", calibration), *groups.items()]:
        values = model.predict([r["input"] for r in rows])
        assert isinstance(values, np.ndarray) and values.shape == (len(rows),)
        assert values.dtype.kind in "iuf" and np.isfinite(values).all()
        metrics["max_absolute_error"][name] = float(max(abs(values - [r["value"] for r in rows])))
    log = Path("/logs/verifier/metrics.json")
    if log.parent.is_dir():
        log.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def test_fit(errors):
    assert .75 <= errors["rate"] <= 1.15
    assert errors["max_absolute_error"]["calibration"] < 2e-5


@pytest.mark.parametrize(("group", "tolerance"), [("separate", .003), ("nucleated", .003),
                                               ("joint", .003), ("slit", .015)])
def test_image(errors, group, tolerance):
    assert errors["max_absolute_error"][group] < tolerance
