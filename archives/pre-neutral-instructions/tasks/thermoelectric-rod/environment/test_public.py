from pathlib import Path
import numpy as np
from model import Model


def test_calibration():
    data = dict(np.load(Path(__file__).parent/'data/calibration.npz'))
    model = Model()
    assert model.fit(data) is model
    assert model.conductivity > 0
    residuals = []
    for i, initial in enumerate(data['initial']):
        result = model.predict(data['t'], data['x'], initial, data['current'][i], data['boundary'][i])
        for field in ['temperature', 'voltage']:
            value = result[field]
            assert isinstance(value, np.ndarray) and value.shape == data[field][i].shape
            assert np.isfinite(value).all()
            residuals.extend(((value-data[field][i])/data['sigma_'+field][i]).ravel())
    assert np.sum(np.square(residuals))/(len(residuals)-1) < 1.5
