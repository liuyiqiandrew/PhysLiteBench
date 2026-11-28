import json
from pathlib import Path

import numpy as np

from model import Model


def test_calibration():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(rows) is model and 0.65 <= model.visibility <= 0.95
    for inputs in ([], [r["input"] for r in rows]):
        predictions = model.predict(inputs)
        assert isinstance(predictions, np.ndarray) and predictions.shape == (len(inputs),)
        assert predictions.dtype.kind in "iuf" and np.isfinite(predictions).all()
    assert max(abs(predictions - [r["value"] for r in rows])) < 2e-6
