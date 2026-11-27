import json
from pathlib import Path

import numpy as np

from model import Model


def test_calibration():
    records = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(records) is model
    assert 0.65 <= model.visibility <= 0.95
    assert model.predict([]).shape == (0,)
    y = model.predict([r["input"] for r in records])
    assert isinstance(y, np.ndarray) and y.shape == (len(records),)
    residual = (y - [r["value"] for r in records]) / [r["sigma"] for r in records]
    assert np.mean(residual**2) < 2.5
