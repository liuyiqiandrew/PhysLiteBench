import json
from pathlib import Path
import numpy as np
from model import Model

TOL = 0.004


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
    first = dict(
        launch=0.0, landing=1.5, heights=[0.0, 1.0], prior=1.0, window=[0.1, 0.65]
    )
    second = dict(
        launch=0.32, landing=1.4, heights=[0.35, 1.1], prior=0.76, window=[0.07, 0.85]
    )
    return [
        (first, [0.37593348, 0.65709984, 0.79144132]),
        (second, [0.50750033, 0.67360928, 0.91330529]),
    ]


def test_worked_readings():
    model = Model().fit(
        [dict(input=dict(mode="free", wave_number=1.0, lag=1.0), value=np.exp(-0.89))]
    )
    for q, expected in examples():
        values = model.predict(
            [dict(q, observable=k) for k in ("final", "history", "duration")]
        )
        assert np.max(np.abs(values - expected)) < 0.004
