import json
from pathlib import Path
import numpy as np
from model import Model

def test_interface_and_calibration():
    records=json.loads((Path(__file__).parent/'data/calibration.json').read_text())
    model=Model();assert model.fit(records) is model
    predicted=model.predict([r['input'] for r in records])
    assert isinstance(predicted,np.ndarray) and predicted.shape==(len(records),) and np.isfinite(predicted).all()
    residual=(predicted-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    assert np.mean(residual**2)<1.5
