import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration():
    records = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(records) is model and .4 <= model.intensity <= 1.3
    assert model.predict([]).shape == (0,)
    y = model.predict([r["input"] for r in records])
    assert isinstance(y,np.ndarray) and y.shape == (len(records),)
    assert np.max(abs(y - [r["value"] for r in records])) < 1e-4
