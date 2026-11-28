import json
from pathlib import Path
import numpy as np
from model import Model

TOL = np.array([0.006, 0.004, 0.004])


def test_fit():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(rows) is model
    assert 0.75 <= model.diffusivity <= 1.2
    for queries in ([], [r["input"] for r in rows]):
        pred = model.predict(queries)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (len(queries),) and pred.dtype.kind in "iuf"
        if queries:
            assert np.max(np.abs(pred - [r["value"] for r in rows])) < 0.00002


def examples():
    q = dict(
        islands=[[-1.0, 1.0]],
        launch=0.0,
        landing=2.0,
        donor=dict(island=0, profile=[1.0, 1 / 3, 1 / 9]),
        rates=[0.83],
        relay=0.7,
    )
    paired = dict(
        q,
        islands=[[-1.0, 1.0], [1.4, 2.8]],
        landing=1.16,
        donor=dict(island=1, profile=[0.9, 0.3, 0.1]),
        rates=[0.8, 0.33],
    )
    return [
        (q, [1 / (2 * np.pi * np.sqrt(3)), 2 * np.sqrt(3) / 9, 0.54381329]),
        (paired, [0.53586148, 0.07038923, 0.31554063]),
    ]


def test_worked_readings():
    model = Model().fit(
        [dict(input=dict(mode="free", wave_number=1.0, lag=1.0), value=np.exp(-0.91))]
    )
    for q, want in examples():
        pred = model.predict(
            [dict(q, observable=k) for k in ("landing", "donor", "clock")]
        )
        assert np.all(np.abs(pred - want) < TOL)
