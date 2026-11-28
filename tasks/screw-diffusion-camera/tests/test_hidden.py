import json
from pathlib import Path

import numpy as np
import pytest

from model import Model


@pytest.fixture(scope="module")
def metrics():
    folder = Path(__file__).parent/"data"
    rows = json.loads((folder/"calibration.json").read_text())
    groups = json.loads((folder/"hidden.json").read_text())
    model = Model()
    assert model.fit(rows) is model
    errors = {}
    for name, group in [("calibration",rows), *groups.items()]:
        result = model.predict([r["input"] for r in group])
        assert isinstance(result,np.ndarray) and result.shape==(len(group),)
        assert result.dtype.kind in "iuf" and np.isfinite(result).all()
        errors[name] = float(max(abs(result-[r["value"] for r in group])))
    record = {"mobility":float(model.mobility),"max_absolute_error":errors}
    path = Path("/logs/verifier/metrics.json")
    if path.parent.is_dir():
        path.write_text(json.dumps(record,indent=2)+"\n")
    return record


def test_fit(metrics):
    assert .7 <= metrics["mobility"] <= 1.2
    assert metrics["max_absolute_error"]["calibration"] < 2e-6


@pytest.mark.parametrize("family",["origin","tagged","quadrature","reversal","current","phase_lock","return_plus","return_minus","return_contrast","commuting","crossed","exchange","lab_echo"])
def test_fringe(metrics, family):
    assert metrics["max_absolute_error"][family] < .003
