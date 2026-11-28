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
    lam = np.log(2.0)
    bell = [np.pi / 4, 0.0, 0.0]
    q = dict(
        loading=lam,
        parity=0,
        marker=dict(sign=1, phase=0.0),
        outer=[bell, bell],
        bank=[[0, 0, 0], [np.pi / 2, 0, 0]],
        analyzers=[0, 0],
        report=0,
        spin=["XX", "XX"],
    )
    product = dict(q, bank=[[0.0, np.pi / 2, 0.0]] * 2, spin=["ZX", "XZ"])
    return [
        (q, [0.5, 0.125, 2 * lam**2]),
        (dict(q, parity=1), [0.5, 0.125, 0.0]),
        (product, [1.0, 0.125, 1.0]),
        (dict(product, report=1), [1.0, 0.125, -1.0]),
    ]


def test_two_swaps():
    model = Model().fit([dict(input=dict(exposure=1.0), value=np.exp(-1.0))])
    for q, expected in examples():
        inputs = [dict(q, observable=k) for k in ("marker", "herald", "fringe")]
        assert np.max(abs(model.predict(inputs) - expected)) < 0.003
