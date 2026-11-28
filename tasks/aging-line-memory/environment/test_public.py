import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    m = Model()
    assert m.fit(rows) is m and 0.9 <= m.depth_scale <= 1.1
    for inputs in ([], [r["input"] for r in rows]):
        pred = m.predict(inputs)
        assert isinstance(pred, np.ndarray) and pred.shape == (len(inputs),)
        assert pred.dtype.kind in "iuf" and np.isfinite(pred).all()
        if inputs:
            assert max(abs(pred - [r["value"] for r in rows])) < 1e-5
