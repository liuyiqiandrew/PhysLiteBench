import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from model import Model

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('physical_reference', HERE/'reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
METADATA = json.loads((HERE/'metadata.json').read_text())


@pytest.fixture(scope='module')
def evaluated():
    records = json.loads((HERE/'data/calibration.json').read_text())
    model = Model()
    assert model.fit(records) is model
    parameter = float(model.drag)
    y = model.predict([r['input'] for r in records])
    assert isinstance(y, np.ndarray) and y.shape == (len(records),)
    assert np.isfinite(y).all()
    residual = (y-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    result = {'parameter': parameter,
              'parameter_relative_error': abs(parameter/reference.TRUE_PARAMETER-1),
              'calibration_chi2': float(residual@residual)/(len(records)-1), 'hidden': {}}
    for name, experiments in reference.hidden_inputs().items():
        truth = reference.predict(experiments, reference.TRUE_PARAMETER)
        actual = model.predict(experiments)
        assert isinstance(actual, np.ndarray) and actual.shape == truth.shape
        assert np.isfinite(actual).all()
        result['hidden'][name] = float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))
    log = Path('/logs/verifier/metrics.json')
    if log.parent.exists():
        log.write_text(json.dumps(result, indent=2)+'\n')
    return result


def test_private_calibration(evaluated):
    assert evaluated['calibration_chi2'] < 1.5


def test_parameter_recovery(evaluated):
    assert evaluated['parameter_relative_error'] < .03


@pytest.mark.parametrize('case', list(reference.hidden_inputs()))
def test_prediction(evaluated, case):
    assert evaluated['hidden'][case] < METADATA['prediction_limit']
