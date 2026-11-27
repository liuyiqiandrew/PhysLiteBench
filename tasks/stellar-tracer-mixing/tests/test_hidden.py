import json
from pathlib import Path

import numpy as np
import pytest
from model import Model

DATA = Path(__file__).parent/"data"


@pytest.fixture(scope="module")
def metrics():
    rows = json.loads((DATA/"calibration.json").read_text())
    truth = json.loads((DATA/"predictions.json").read_text())
    model = Model()
    assert model.fit(rows) is model
    errors = {}
    for group, measurements in [("calibration",rows),*truth.items()]:
        y = model.predict([r["input"] for r in measurements])
        target = np.array([r["value"] for r in measurements])
        assert isinstance(y,np.ndarray) and y.shape == target.shape and np.all(np.isfinite(y))
        errors[group] = float(np.max(abs(y-target)))
    result = {"dispersion":float(model.dispersion),"max_absolute_error":errors}
    path=Path("/logs/verifier/metrics.json")
    if path.parent.exists():
        path.write_text(json.dumps(result,indent=2)+"\n")
    return result


def test_fit(metrics):
    assert abs(metrics["dispersion"]-.947)<.001
    assert metrics["max_absolute_error"]["calibration"] < 2e-5


@pytest.mark.parametrize("group",["joint_density","transferred_echo","spectral_band"])
def test_echo(metrics,group):
    assert metrics["max_absolute_error"][group] < .00045
