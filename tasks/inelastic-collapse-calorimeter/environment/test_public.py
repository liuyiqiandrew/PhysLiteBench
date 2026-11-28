import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    m = Model()
    assert m.fit(rows) is m and 0.72 <= m.diffusivity <= 1.10
    for batch in ([], [r["input"] for r in rows]):
        prediction = m.predict(batch)
        assert isinstance(prediction, np.ndarray) and prediction.shape == (len(batch),)
        assert prediction.dtype.kind in "iuf" and np.isfinite(prediction).all()
    assert np.max(abs(prediction - [r["value"] for r in rows])) < 2e-6
