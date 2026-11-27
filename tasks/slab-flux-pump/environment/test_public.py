import json
from pathlib import Path

import numpy as np
from model import Model


def test_fit_and_predict():
    data = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(data) is model
    assert 0.85 <= model.pinning <= 1.15
    inputs = [r["input"] for r in data]
    result = model.predict(inputs)
    assert isinstance(result, np.ndarray) and result.shape == (len(inputs),)
    np.testing.assert_allclose(result, [r["value"] for r in data], atol=1e-5, rtol=0)
    assert model.predict([]).shape == (0,)
