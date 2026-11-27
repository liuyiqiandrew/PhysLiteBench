import json
from pathlib import Path

import numpy as np

from model import Model


def test_calibration():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(rows) is model
    assert 2 <= model.exponent <= 4
    values = model.predict([r["input"] for r in rows])
    assert isinstance(values, np.ndarray)
    assert values.shape == (len(rows),)
    np.testing.assert_allclose(values, [r["value"] for r in rows], rtol=0.002)


def test_aperture():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model().fit(rows)
    q = dict(slow=[1, 1], steps=[], peri=2, apo=9,
             train=dict(coupling=[1, 0], exposure=.2, binding_limit=.21))
    np.testing.assert_allclose(model.predict([q]), [0], atol=1e-10)
