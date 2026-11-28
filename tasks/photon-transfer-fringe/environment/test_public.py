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


def test_separated_raman_pair():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    model = Model().fit(rows)
    modes = [[[1, 0], [0, 0]], [[0, 0], [1, 0]]]
    q = dict(
        rates=[1, 1],
        durations=[1, 1],
        detuning=[0, 0],
        atom=[0, 0],
        analyzer=None,
        control=[1, 0, 0, 0],
        input_modes=modes,
        output_modes=modes,
        input_rotation=[0, 0],
        output_rotations=[[0, 0], [0, 0]],
        phase=0,
        **{"lambda": np.pi}
    )
    assert abs(model.predict([q])[0] - 0.7708459436069642 * model.visibility) < 0.001
