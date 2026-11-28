import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration_and_label_symmetry():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    m = Model()
    assert m.fit(rows) is m and 0.75 <= m.activity <= 1.15
    p = m.predict([r["input"] for r in rows])
    assert (
        p.dtype.kind in "ifu"
        and p.shape == (len(rows),)
        and m.predict([]).shape == (0,)
    )
    assert max(abs(p - [r["value"] for r in rows])) < 0.00002
    chamber = dict(feed=4, routes=[1, 1, 1, 1])
    q = dict(
        trunk=chamber,
        arms=[chamber, chamber],
        gate=[0, 1, 2, 3],
        shifted=True,
        length=3,
    )
    pred = m.predict([dict(q, probe=[i]) for i in range(4)])
    np.testing.assert_allclose(pred, 0.25, atol=0.004, rtol=0)
