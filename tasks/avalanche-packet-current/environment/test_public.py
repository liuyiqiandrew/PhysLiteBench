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


def test_prompt_and_delayed_packets():
    m, _ = fitted()
    a = m.activity
    q = dict(
        depth=0.2,
        velocities=[1.0, 1.0],
        dead_space=[0.45, 0.12],
        ionization=[1.0, 0.0],
        feedback=0.0,
        gain=2,
        last_delay=0.0,
        multiplicity=2,
        tag=None,
        read=[0.0, 0.8],
    )
    mean = 0.45 + 1 / a - 0.35 / np.expm1(0.35 * a)
    assert abs(m.predict([q])[0] - (1 + 2 * (0.8 - mean)) / 2) < 0.003
    q.update(
        depth=0.4,
        dead_space=[0.05, 0.05],
        ionization=[2.0, 2.0],
        feedback=0.6,
        last_delay=0.3,
        multiplicity=1,
        read=[0.22, 0.7],
    )
    assert abs(m.predict([q])[0] - 0.61) < 0.003
