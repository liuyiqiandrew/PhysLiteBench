from pathlib import Path
import numpy as np
import pytest
from model import TransportModel

@pytest.fixture(scope="module")
def data():
    return dict(np.load(Path(__file__).parent / "data/calibration.npz"))

@pytest.fixture(scope="module")
def fitted(data):
    model = TransportModel()
    assert model.fit(data) is model
    return model

def test_parameter(fitted):
    assert np.isfinite(fitted.diffusivity) and fitted.diffusivity > 0

def test_calibration(fitted, data):
    predictions = []
    for initial in data["initial"]:
        predicted = fitted.predict(data["t"], data["x"], initial)
        assert isinstance(predicted, np.ndarray)
        assert predicted.shape == data["concentration"].shape[1:]
        assert np.isfinite(predicted).all()
        predictions.append(predicted)
    residual = (np.array(predictions)-data["concentration"])/data["sigma"]
    assert np.sum(residual**2)/(residual.size-1) < 1.5


def test_correlation_interface():
    from pathlib import Path
    import numpy as np
    from model import TransportModel
    data=dict(np.load(Path(__file__).parent/'data/calibration.npz'))
    model=TransportModel().fit(data)
    result=model.correlate(np.array([0.,1.,10.]),np.array([.8,1.2]),2)
    assert isinstance(result,np.ndarray) and result.shape==(3,2,2)
    assert np.isfinite(result).all()
