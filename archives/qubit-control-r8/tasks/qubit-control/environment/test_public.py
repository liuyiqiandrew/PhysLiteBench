import json
from pathlib import Path
import numpy as np
import pytest
from model import QubitModel


@pytest.fixture(scope='module')
def fitted():
    records=json.loads((Path(__file__).parent/'data/calibration.json').read_text())
    model=QubitModel()
    assert model.fit(records) is model
    return model,records


def test_interface(fitted):
    model,records=fitted
    assert np.isfinite(model.gamma) and .04<=model.gamma<=.25
    values=model.predict([r['experiment'] for r in records])
    assert isinstance(values,np.ndarray) and values.shape==(len(records),)
    assert np.isfinite(values).all() and np.all(values>=-1e-12) and np.all(values<=1+1e-12)


def test_calibration(fitted):
    model,records=fitted
    values=model.predict([r['experiment'] for r in records])
    residual=(values-np.array([r['probability'] for r in records]))/np.array([r['sigma'] for r in records])
    assert float(residual@residual)/(len(records)-1)<1.5
