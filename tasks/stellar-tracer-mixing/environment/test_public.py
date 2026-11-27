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
