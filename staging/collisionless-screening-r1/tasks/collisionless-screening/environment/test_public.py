import json
from pathlib import Path
import numpy as np
from model import Model


def test_public_calibration():
    rows=json.loads((Path(__file__).parent/'data/calibration.json').read_text())
    model=Model()
    assert model.fit(rows) is model
    prediction=model.predict([row['input'] for row in rows])
    assert isinstance(prediction,np.ndarray) and prediction.shape==(len(rows),)
    assert np.isfinite(prediction).all()
    residual=(prediction-np.asarray([r['value'] for r in rows]))/np.asarray([r['sigma'] for r in rows])
    assert np.mean(residual**2)<1.5
