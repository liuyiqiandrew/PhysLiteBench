import json
from pathlib import Path
import numpy as np
from model import Model


def fitted():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    m = Model()
    assert m.fit(rows) is m
    return m, rows


def test_calibration():
    m, rows = fitted()
    assert 0.75 <= m.activity <= 1.15
    x = m.predict([])
    assert isinstance(x, np.ndarray) and x.shape == (0,) and x.dtype.kind in "iuf"
    y = m.predict([r["input"] for r in rows])
    assert (
        isinstance(y, np.ndarray) and y.shape == (len(rows),) and y.dtype.kind in "iuf"
    )
    assert np.max(np.abs(y - [r["value"] for r in rows])) < 0.00002


def examples():
    t = np.log(2.0)
    single = dict(
        links=[[0, 1, t, t, -0.35]], parities=[1, 1], requests=[[0, 1], [0, 1]]
    )
    first = [3 / 5 - 8 * t / 15, 8 * t / 9, 1.0]
    # Capacities of the direct path D and the two-link bypass L, clipped at two.
    p = np.array([0.5, t / 2, (1 - t) / 2])
    l = np.array([0.75, 0.25 - p[2] ** 2, p[2] ** 2])
    joint = fringe = 0.0
    w = -0.24
    for d in range(3):
        for b in range(3):
            count = int(d == 2) + int(b == 2) + 2 * int(d > 0 and b > 0)
            if count:
                joint += p[d] * l[b]
                fringe += (
                    p[d]
                    * l[b]
                    * (int(d == 2) + int(b == 2) + 2 * w * int(d > 0 and b > 0))
                    / count
                )
    second = [joint, 1 - joint / (5 / 8), fringe / joint]
    tri = dict(
        links=[[0, 2, t, 0, 0.5], [0, 1, t, 0, 0.6], [1, 2, t, 0, -0.8]],
        parities=[None] * 3,
        requests=[[0, 2], [0, 2]],
    )
    return [
        (dict(q, observable=k), v)
        for q, truth in [(single, first), (tri, second)]
        for k, v in zip(("delivery", "contention", "fringe"), truth)
    ]


def test_one_link_and_triangle():
    model = Model().fit([dict(input=dict(exposure=1.0), value=np.exp(-1.0))])
    for q, value in examples():
        assert abs(model.predict([q])[0] - value) < 0.003
