import json
from pathlib import Path

import numpy as np
from model import Model


def test_calibration_and_shapes():
    data = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(data) is model
    assert 0.55 <= model.width <= 1.45
    predicted = model.predict_calibration(data)
    values = np.array([row["curvature"] for row in data])
    np.testing.assert_allclose(predicted, values, rtol=0, atol=1e-5)
    scan = {"start": 0, "signs": [1, -1], "hops": [0.5]}
    pair = {"mode": "covariance", "gain": 0.5, "scan": scan, "partner": scan}
    echo = {"mode": "echo", "gain": 0, "scan": scan}
    result = model.predict([echo, pair, echo])
    assert result.shape == (3,) and np.isfinite(result).all()
    np.testing.assert_allclose(result[[0, 2]], 1, atol=1e-9)
    assert model.predict([]).shape == (0,)
