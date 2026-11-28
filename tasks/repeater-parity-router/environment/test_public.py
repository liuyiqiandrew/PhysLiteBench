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
    a = np.log(2.0)
    single = dict(
        links=[[0, 1, a, a, -0.35]],
        parities=[1, 1],
        requests=[[0, 1], [0, 1]],
        pilot=dict(links=[0], count=3),
    )
    tri = dict(
        links=[[0, 2, a, 0, 0.5], [0, 1, a, 0, 0.6], [1, 2, a, 0, -0.8]],
        parities=[None] * 3,
        requests=[[0, 2], [0, 2]],
        pilot=dict(links=[0], count=1),
    )
    return [
        (dict(q, observable=k), v)
        for q, vs in [
            (single, [0.875, 0.0, 1.0]),
            (tri, [0.25, 0.75, 0.48 + 0.52 / 3 * (1 - a) ** 2]),
        ]
        for k, v in zip(("delivery", "contention", "fringe"), vs)
    ]


def test_one_link_and_triangle():
    model = Model().fit([dict(input=dict(exposure=1.0), value=np.exp(-1.0))])
    for q, value in examples():
        assert abs(model.predict([q])[0] - value) < 0.003
