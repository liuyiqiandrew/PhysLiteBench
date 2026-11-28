import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration_and_exchange():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    m = Model()
    assert m.fit(rows) is m and 0.75 <= m.gain <= 1.15
    p = m.predict([r["input"] for r in rows])
    assert (
        p.dtype.kind in "ifu"
        and p.shape == (len(rows),)
        and m.predict([]).shape == (0,)
    )
    assert max(abs(p - [r["value"] for r in rows])) < 0.00002
    a = dict(aspect=0.75, marks=[[0.25, 0.25], [0.625, 0.75]])
    b = dict(
        aspect=1.0, marks=[[0.125, 0.25], [0.25, 0.5], [0.625, 0.25], [0.75, 0.75]]
    )
    q = dict(films=[a, b], bias=[0.35, -0.2], coupling=0.4, read="overlap")
    values = m.predict([q, dict(q, films=[b, a])])
    assert (
        values.shape == (2,)
        and np.isfinite(values).all()
        and abs(values[0] - values[1]) < 0.004
    )
