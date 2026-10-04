import json
from pathlib import Path
import numpy as np
import pytest
from model import Model


@pytest.fixture(scope='module')
def records():
    return json.loads((Path(__file__).parent/'data/calibration.json').read_text())


@pytest.fixture(scope='module')
def fitted(records):
    model = Model()
    assert model.fit(records) is model
    return model


def test_interface(fitted, records):
    value = fitted.viscosity
    assert np.isfinite(value) and 0.004 <= value <= 0.016
    y = fitted.predict([r['input'] for r in records])
    assert isinstance(y, np.ndarray) and y.shape == (len(records),)
    assert np.isfinite(y).all()


def test_calibration(fitted, records):
    y = fitted.predict([r['input'] for r in records])
    residual = (y-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    assert float(residual@residual)/(len(records)-1) < 1.5
