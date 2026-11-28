from pathlib import Path
import json
import numpy as np
from model import Model


def test_fit_and_reversal():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    m = Model()
    assert m.fit(rows) is m and 0.7 <= m.gain <= 1.1
    p = m.predict([r["input"] for r in rows])
    assert (
        isinstance(p, np.ndarray) and p.dtype.kind in "ifu" and p.shape == (len(rows),)
    )
    assert (
        m.predict([]).shape == (0,)
        and max(abs(p - [r["value"] for r in rows])) < 0.000002
    )
    q = dict(
        aspect=2.0,
        shear=0.25,
        bias=[0.2, -0.6],
        gate=[1, 1, 3, 1],
        probe=[0.8, -1.2],
        phase=0.41,
    )
    reverse = dict(q, bias=[-0.2, 0.6], gate=[1, 1, 3, 2], phase=-0.41)
    a, b = m.predict([q, reverse])
    assert np.isfinite([a, b]).all() and abs(a - b) < 0.004
