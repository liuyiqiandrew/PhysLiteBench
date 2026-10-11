import importlib.util
import json
import math
import os
from pathlib import Path
import numpy as np
import pytest
from model import Model

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('physical_reference', HERE/'reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
# reference.predict(hidden_inputs, TRUE_PARAMETER), frozen by scripts/freeze_hidden_truth.py.
FROZEN = json.loads((HERE/'data/hidden_truth.json').read_text())['groups']


def canonical(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x))


def same_inputs(a, b):
    """Equal structure and strings; numbers equal up to last-bit differences between CPUs."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_inputs(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(same_inputs, a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-300)
    return a == b


@pytest.fixture(scope='module')
def evaluated():
    records = json.loads((HERE/'data/calibration.json').read_text())
    model = Model()
    assert model.fit(records) is model
    fitted = float(getattr(model, reference.PARAMETER))
    y = model.predict([r['input'] for r in records])
    residual = (y-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    result = {'parameter': fitted, 'parameter_relative_error': abs(fitted/reference.TRUE_PARAMETER-1),
              'calibration_chi2': float(residual@residual)/(len(records)-1), 'hidden': {}}
    for name, experiments in reference.hidden_inputs().items():
        assert same_inputs(FROZEN[name]['inputs'], canonical(experiments))
        truth = np.array(FROZEN[name]['truth'])
        actual = model.predict(experiments)
        assert isinstance(actual, np.ndarray) and actual.shape == truth.shape
        assert np.isfinite(actual).all()
        scale = np.full_like(truth, np.sqrt(np.mean(truth**2)))
        result['hidden'][name] = float(np.sqrt(np.mean(((actual-truth)/scale)**2)))
    log = Path(os.environ.get('METRICS_PATH', '/logs/verifier/metrics.json'))
    if log.parent.exists():
        log.write_text(json.dumps(result, indent=2)+'\n')
    return result


def test_private_calibration(evaluated):
    assert evaluated['calibration_chi2'] < 1.5


def test_parameter_recovery(evaluated):
    assert evaluated['parameter_relative_error'] < .03


@pytest.mark.parametrize('case', list(reference.hidden_inputs()))
def test_prediction(evaluated, case):
    assert evaluated['hidden'][case] < 0.04
