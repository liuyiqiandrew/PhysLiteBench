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
    assert 0.8 <= m.activity <= 1.1
    x = m.predict([])
    assert isinstance(x, np.ndarray) and x.shape == (0,) and x.dtype.kind in "iuf"
    y = m.predict([r["input"] for r in rows])
    assert (
        isinstance(y, np.ndarray) and y.shape == (len(rows),) and y.dtype.kind in "iuf"
    )
    assert np.max(np.abs(y - [r["value"] for r in rows])) < 0.00002


def examples():
    q = dict(
        depth=0.4,
        velocities=[1.0, 1.0],
        ionization=[2.0, 2.0],
        dead_space=[0.0, 0.0],
        feedback=0.0,
        gain=3,
        tag=None,
        last_delays=[0.0, 0.6],
        multiplicities=[None, None],
        read=[0.0, 0.25],
        gates=[[0.0, 0.15], [0.0, 0.23]],
    )
    results = []
    for part, z, intensity, current in [
        ("point", [0.3, 0.46], 7 / 30, 59 / 350),
        ("ridge", [0.3, 0.54], 5 / 3, 31 / 150),
        ("ridge", [0.45, 0.77], 5 / (3 * np.sqrt(2)), 17 / 60),
        ("area", [0.31, 0.83], 50 / 9, 113 / 400),
    ]:
        if part == "area":
            q.update(gain=4, gates=[[0.0, 0.1], [0.0, 0.2]])
        results.append((dict(q, part=part, charges=z), intensity, current))
    return results


def test_joint_charge_spectrum():
    m, _ = fitted()
    for q, i, w in examples():
        tol = {"point": 0.004, "ridge": 0.015, "area": 0.03}[q["part"]]
        assert abs(m.predict([dict(q, observable="intensity")])[0] - i) < tol
        assert abs(m.predict([dict(q, observable="waveform")])[0] - w) < 0.004
