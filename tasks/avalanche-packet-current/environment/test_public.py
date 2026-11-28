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


def worked():
    return dict(
        depth=0.4,
        velocities=[1.0, 1.0],
        ionization=[2.0, 2.0],
        dead_space=[0.0, 0.0],
        feedback=0.0,
        gain=3,
        tag=None,
        read=[0.25, 0.7],
        last_delays=[0.3, 0.4],
        multiplicities=[2, 1],
        observable="waveform",
    )


def test_paired_arrivals():
    model, _ = fitted()
    q = worked()
    for pair, val in [([2, 1], 0.6), ([1, 2], 2 / 3), ([1, 1], 19 / 30)]:
        q["multiplicities"] = pair
        q["observable"] = "waveform"
        assert abs(model.predict([q])[0] - val) < 0.004
        q["observable"] = "fraction"
        assert abs(model.predict([q])[0] - 1 / 3) < 0.004
