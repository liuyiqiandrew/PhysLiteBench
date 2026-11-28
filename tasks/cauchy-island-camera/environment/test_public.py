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
    pair = dict(
        source="pair",
        launch=0.0,
        recoil=0.30,
        rate=0.0,
        thresholds=[0.25, 0.25],
        efficiencies=[1.0, 1.0],
    )
    single = dict(
        source="single",
        launch=0.23,
        recoil=0.31,
        rate=0.32,
        thresholds=[0.18, 0.42],
        efficiencies=[0.8, 0.9],
    )
    return [
        (pair, [0.49435347, 0.34109898, 1.0]),
        (single, [0.29043, 0.15084, 0.66374]),
    ]


def test_worked_readings():
    model = Model().fit(
        [dict(input=dict(mode="free", wave_number=1.0, lag=1.0), value=np.exp(-0.89))]
    )
    for q, expected in examples():
        values = model.predict(
            [dict(q, observable=k) for k in ("red", "coincidence", "clean")]
        )
        assert np.max(np.abs(values - expected)) < 0.004
