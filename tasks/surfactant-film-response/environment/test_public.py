import json
from pathlib import Path

import numpy as np

from model import Model


def test_camera():
    rows=json.loads((Path(__file__).parent/'data/calibration.json').read_text())
    model=Model()
    assert model.fit(rows) is model
    assert .85 <= model.rate <= 1.15
    assert model.predict([]).shape == (0,)
    y=model.predict([r['input'] for r in rows])
    assert isinstance(y,np.ndarray) and y.shape==(len(rows),)
    assert y.dtype.kind in 'iuf' and np.isfinite(y).all()
    assert max(abs(y-[r['value'] for r in rows])) < 2e-5
