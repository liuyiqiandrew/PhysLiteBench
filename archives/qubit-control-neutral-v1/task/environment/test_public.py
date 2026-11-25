import json
from pathlib import Path
import numpy as np
import pytest
from model import QubitModel

@pytest.fixture(scope="module")
def fitted():
    runs = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = QubitModel()
    assert model.fit(runs) is model
    return model, runs

def test_gamma(fitted):
    assert np.isfinite(fitted[0].gamma) and fitted[0].gamma > 0

def test_output(fitted):
    model, runs = fitted
    p = model.predict([r["experiment"] for r in runs])
    assert isinstance(p, np.ndarray) and p.shape == (len(runs),)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))

def test_calibration(fitted):
    model, runs = fitted
    p = model.predict([r["experiment"] for r in runs])
    chi2 = sum(np.sum(((v-r["probability"])/r["sigma"])**2) for v,r in zip(p,runs))
    assert chi2/(len(runs)-1) < 1.5
