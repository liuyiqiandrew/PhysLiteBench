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


def test_rotating_probes_are_finite():
    data = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model().fit(data)
    q = dict(amplitude=.19, normal=.27, cutting=.7, conductance=.8,
             hall=.4, coating=.06, sense=1)
    cases = [dict(q, readout="induction", axis="y", depth=0.),
             dict(q, readout="normal_voltage", depth=0.),
             dict(q, readout="sheet_power")]
    result = model.predict(cases)
    assert result.shape == (3,) and np.isfinite(result).all()
    np.testing.assert_allclose(result[:2], [.19, 0.], atol=1e-7)
    assert result[2] >= 0.
