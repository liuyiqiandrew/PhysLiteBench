from pathlib import Path
import json

import numpy as np

from model import Model


def test_interface_and_calibration():
    records = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(records) is model
    values = model.predict([row["input"] for row in records])
    assert isinstance(values, np.ndarray) and values.shape == (len(records),)
    assert np.isfinite(values).all()
    residual = (values - np.array([r["value"] for r in records])) / [r["sigma"] for r in records]
    assert residual @ residual / (len(records) - 1) < 1.5
    assert model.predict([]).shape == (0,)
