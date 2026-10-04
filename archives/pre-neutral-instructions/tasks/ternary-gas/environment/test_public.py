from pathlib import Path
import numpy as np
from model import Model


def test_calibration():
    data = dict(np.load(Path(__file__).parent/'data/calibration.npz'))
    model = Model()
    assert model.fit(data) is model
    assert np.isfinite(model.diffusivity) and model.diffusivity > 0
    prediction = np.array([model.predict(data['t'], data['x'], initial)
                           for initial in data['initial']])
    assert prediction.shape == data['mole_fraction'].shape
    assert np.isfinite(prediction).all()
    residual = (prediction-data['mole_fraction'])/data['sigma']
    assert np.sum(residual**2)/(residual.size-1) < 1.5
