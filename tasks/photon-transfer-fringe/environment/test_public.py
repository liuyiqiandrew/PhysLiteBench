import json
from pathlib import Path

import numpy as np
from model import Model


def test_calibration():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model()
    assert model.fit(rows) is model and 0.65 <= model.visibility <= 0.95
    empty = model.predict([])
    assert (
        isinstance(empty, np.ndarray)
        and empty.shape == (0,)
        and empty.dtype.kind in "iuf"
    )
    values = model.predict([row["input"] for row in rows])
    assert isinstance(values, np.ndarray) and values.shape == (len(rows),)
    assert values.dtype.kind in "iuf" and np.all(np.isfinite(values))
    assert max(abs(values - [row["value"] for row in rows])) < 0.00002


def square():
    # Both polarizations occupy the whole square pulse. Polarization 1 is a spectator.
    flat = [[1, 0]]
    return dict(
        rates=[1.0, 0.0],
        atom=[0, 0],
        analyzer=None,
        control=[0, 0, 0],
        gap=[0, 0],
        first=dict(
            durations=[1.0],
            detuning=[0.0],
            input_modes=[flat, flat],
            input_rotation=[0, 0],
            write_mode=flat,
        ),
        second=dict(
            durations=[1.0],
            detuning=[0.0],
            read_mode=flat,
            output_mode=flat,
            output_rotation=[0, 0],
        ),
        store=dict(
            cells=2,
            write_k=[0.0, 0.0],
            read_k=[0.0, 0.0],
            write=[np.pi / (2 * np.sqrt(2))] * 2,
            read=[np.pi / (2 * np.sqrt(2))] * 2,
            rotation=[0, 0],
            phase=0.0,
        ),
        handoff="unmeasured",
        conditional=False,
        phase=0.0,
        **{"lambda": 0.0}
    )


def test_square_echo():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model().fit(rows)
    q = square()
    for phi, value in [(0.0, 0.7868098728458969), (np.pi, 0.1235676703171777)]:
        q["store"]["phase"] = phi
        assert abs(model.predict([q])[0] - value * model.visibility) < 0.001
