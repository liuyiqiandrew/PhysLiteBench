import json
from pathlib import Path

import numpy as np
from model import Model


def test_calibration():
    rows = json.loads((Path(__file__).parent/'data/calibration.json').read_text())
    model = Model()
    assert model.fit(rows) is model and .65 <= model.visibility <= .95
    empty = model.predict([])
    assert isinstance(empty, np.ndarray) and empty.shape == (0,) and empty.dtype.kind in 'iuf'
    values = model.predict([row['input'] for row in rows])
    assert isinstance(values,np.ndarray) and values.shape == (len(rows),)
    assert values.dtype.kind in 'iuf' and np.all(np.isfinite(values))
    assert max(abs(values-[row['value'] for row in rows])) < .00002
