import json
from pathlib import Path

import numpy as np
import pytest

from model import Model


@pytest.fixture(scope="module")
def metrics():
    folder = Path(__file__).parent / "data"
    rows = json.loads((folder / "calibration.json").read_text())
    model = Model()
    assert model.fit(rows) is model
    errors = {}
    groups = json.loads((folder / "hidden.json").read_text())
    for name, records in [("calibration", rows), *groups.items()]:
        values = model.predict([r["input"] for r in records])
        assert isinstance(values, np.ndarray) and values.shape == (len(records),)
        assert values.dtype.kind in "iuf" and np.isfinite(values).all()
        errors[name] = float(np.max(abs(values - [r["value"] for r in records])))
    result = {"visibility": float(model.visibility), "max_absolute_error": errors}
    log = Path("/logs/verifier/metrics.json")
    if log.parent.is_dir():
        log.write_text(json.dumps(result, indent=2) + "\n")
    return result


def test_fit(metrics):
    assert .65 <= metrics["visibility"] <= .95
    assert metrics["max_absolute_error"]["calibration"] < 2e-6


@pytest.mark.parametrize("group",["sparse_resonances","overlap_shifts","unresolved_polarity"])
def test_fringe(metrics,group):
    assert metrics["max_absolute_error"][group] < .003
