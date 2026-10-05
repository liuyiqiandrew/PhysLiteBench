import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from model import Model

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('physical_reference',HERE/'reference.py')
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.fixture(scope='module')
def evaluated():
    records=json.loads((HERE/'data/calibration.json').read_text())
    model=Model()
    assert model.fit(records) is model
    fitted=float(model.plasma_frequency)
    actual=model.predict([r['input'] for r in records])
    residual=(actual-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    metrics=dict(parameter=fitted,parameter_relative_error=abs(fitted/reference.TRUE_PARAMETER-1),
                 calibration_chi2=float(residual@residual)/(len(records)-1),hidden={})
    for name,experiments in reference.hidden_inputs().items():
        truth=reference.predict(experiments)
        actual=model.predict(experiments)
        assert isinstance(actual,np.ndarray) and actual.shape==truth.shape
        assert np.isfinite(actual).all()
        metrics['hidden'][name]=float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))
    output=Path('/logs/verifier/metrics.json')
    if output.parent.exists():output.write_text(json.dumps(metrics,indent=2)+'\n')
    return metrics


def test_calibration(evaluated):
    assert evaluated['calibration_chi2']<1.5


def test_parameter(evaluated):
    assert evaluated['parameter_relative_error']<.03


@pytest.mark.parametrize('name',list(reference.hidden_inputs()))
def test_prediction(evaluated,name):
    assert evaluated['hidden'][name]<.04
