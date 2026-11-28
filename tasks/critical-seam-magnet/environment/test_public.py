import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration_and_reference():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    m = Model()
    assert m.fit(rows) is m and 0.75 <= m.gain <= 1.15
    inp = [r["input"] for r in rows]
    pred = m.predict(inp)
    assert pred.dtype.kind in "iuf" and pred.shape == (len(inp),)
    assert (
        m.predict([]).shape == (0,)
        and max(abs(pred - [r["value"] for r in rows])) < 0.00002
    )
    q = dict(
        sites=[0.125, 0.4375], aspect=0.85, bias=0.32, parity=0, seam=1, read="spin"
    )
    np.testing.assert_allclose(
        m.predict([q, dict(q, read="response")]), [1.0, 0.0], atol=0.004, rtol=0
    )
