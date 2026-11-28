import json
from pathlib import Path

import numpy as np

from model import Model


def test_calibration():
    rows = json.loads((Path(__file__).parent/"data/calibration.json").read_text())
    model = Model()
    assert model.fit(rows) is model and .7 <= model.mobility <= 1.2
    for inputs in ([], [r["input"] for r in rows]):
        result = model.predict(inputs)
        assert isinstance(result,np.ndarray) and result.shape == (len(inputs),)
        assert result.dtype.kind in "iuf" and np.isfinite(result).all()
    assert max(abs(result-[r["value"] for r in rows])) < 2e-6
