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


def packed(z):
    return np.stack((z.real, z.imag), axis=-1).tolist()


def square():
    shape = np.ones((2, 1), complex) / np.sqrt(2)
    return dict(
        rates=[1.0, 0.0],
        duration=0.5,
        atom=[0.0, 0.0],
        analyzer=None,
        control=[0.0, 0.0, 0.0],
        gap=[0.0, 0.0],
        input_modes=packed(np.stack((shape, shape))),
        input_rotation=[0.0, 0.0],
        output_mode=packed(shape),
        output_rotation=[0.0, 0.0],
        detuning=[[0.0], [0.0]],
        return_detuning=[[0.0], [0.0]],
        arms=[[0.0, 0.0, np.pi], [0.0, 0.0, np.pi]],
        input_reverse=[False, False],
        output_reverse=[True, True],
        temporal=packed(np.eye(2, dtype=complex)[::-1]),
        phase=0.0,
        conditional=False,
        **{"lambda": 0.0}
    )


def test_square_echo():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model().fit(rows)
    q = square()
    for rev, truth in [(True, 1.0), (False, 0.805758294060817)]:
        q["output_reverse"] = [rev, rev]
        q["temporal"] = packed(np.eye(2, dtype=complex)[:: (-1 if rev else 1)])
        assert abs(model.predict([q])[0] - truth * model.visibility) < 0.001
