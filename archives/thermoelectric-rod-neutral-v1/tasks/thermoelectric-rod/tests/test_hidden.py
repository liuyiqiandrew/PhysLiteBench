import json
from pathlib import Path
import numpy as np
import pytest
from model import Model
from reference import metrics


@pytest.fixture(scope='module')
def evaluated():
    data = dict(np.load(Path(__file__).parent/'data/calibration.npz'))
    result = metrics(Model().fit(data), data)
    output = Path('/logs/verifier/metrics.json')
    if output.parent.exists():
        output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def test_calibration(evaluated):
    assert evaluated['calibration_chi2'] < 1.5


def test_parameter(evaluated):
    assert evaluated['parameter_relative_error'] < .03


def test_voltage(evaluated):
    assert max(evaluated['voltage_absolute_error']) < 1e-5


@pytest.mark.parametrize('i', range(3))
def test_prediction(evaluated, i):
    assert evaluated['hidden'][f'current_{i}'] < .04
