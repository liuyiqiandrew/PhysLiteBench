import json
from pathlib import Path

import numpy as np
from model import Model


def test_calibration():
    rows = json.loads((Path(__file__).parent/"data/calibration.json").read_text())
    model = Model()
    assert model.fit(rows) is model
    assert .8 <= model.dispersion <= 1.2
    values = model.predict([r["input"] for r in rows])
    assert isinstance(values,np.ndarray) and values.shape == (len(rows),)
    assert values.dtype.kind in "iuf"
    assert np.max(abs(values - [r["value"] for r in rows])) < 2e-5
    empty = model.predict([])
    assert isinstance(empty,np.ndarray) and empty.shape == (0,) and empty.dtype.kind in "iuf"


def test_unforced_relay():
    rows = json.loads((Path(__file__).parent/"data/calibration.json").read_text())
    model = Model().fit(rows)
    q = dict(rows[0]["input"], k=[1,3], channel="relay")
    value = model.predict([q])
    assert isinstance(value,np.ndarray) and value.shape == (1,) and value.dtype.kind in "iuf"
    assert abs(value[0]) < .00045
