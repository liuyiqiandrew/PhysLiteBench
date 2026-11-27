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
    np.testing.assert_allclose(values,[r["value"] for r in rows],atol=2e-5,rtol=0)
    assert model.predict([]).shape == (0,)
