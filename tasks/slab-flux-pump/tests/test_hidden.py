import json
from pathlib import Path

import numpy as np
import pytest

from model import Model

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def metrics():
    calibration = json.loads((DATA / "calibration.json").read_text())
    truth = json.loads((DATA / "predictions.json").read_text())
    model = Model()
    assert model.fit(calibration) is model
    errors = {}
    for name, records in [("calibration", calibration), *truth.items()]:
        values = np.array([r["value"] for r in records])
        pred = np.asarray(model.predict([r["input"] for r in records]))
        assert pred.shape == values.shape and np.isfinite(pred).all()
        errors[name] = float(np.max(abs(pred - values)))
    result = {"pinning": float(model.pinning), "max_abs_errors": errors}
    folder = Path("/logs/verifier")
    if folder.is_dir():
        (folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def test_fit(metrics):
    assert abs(metrics["pinning"] - 1.063) < 1e-4
    assert metrics["max_abs_errors"]["calibration"] < 1e-5


@pytest.mark.parametrize("readout,tolerance", [("crest", .003), ("trough", .003), ("impulse", .008),
                                               ("induction", .001), ("electric", .0006),
                                               ("laminate_induction", .001),
                                               ("laminate_electric", .0006),
                                               ("normal_voltage", .00025),
                                               ("sheet_power", .00015),
                                               ("film_induction", .001),
                                               ("film_electric", .0006),
                                               ("film_voltage", .00025),
                                               ("film_power", .00015)])
def test_signal(metrics, readout, tolerance):
    assert metrics["max_abs_errors"][readout] < tolerance
