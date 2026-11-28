import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration():
    records = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(records) is model and .75 <= model.rate <= 1.15
    for rows in ([], records):
        pred = model.predict([r["input"] for r in rows])
        assert isinstance(pred, np.ndarray) and pred.shape == (len(rows),)
        assert pred.dtype.kind in "iuf" and np.isfinite(pred).all()
        if rows:
            assert max(abs(pred - [r["value"] for r in rows])) < 2e-5
