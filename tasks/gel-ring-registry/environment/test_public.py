import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration_and_registration():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    m = Model()
    assert m.fit(rows) is m and 0.75 <= m.activity <= 1.15
    pred = m.predict([r["input"] for r in rows])
    assert pred.shape == (len(rows),) and m.predict([]).shape == (0,)
    assert max(abs(pred - [r["value"] for r in rows])) < 0.00002
    p = dict(feed=[4, 6], routes=[[1] * 4] * 2, switch=[[0.2, 0.3]] * 4, blue=0.45)
    q = dict(length=2, rings=[p, p], gate=[0])
    values = m.predict([dict(q, probe=[0]), dict(q, flux=[np.pi / 2, 0], phase=0)])
    np.testing.assert_allclose(values, [0.6, 0.4], rtol=0, atol=0.004)
