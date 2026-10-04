import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pytest
from model import QubitModel

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('qubit_reference',HERE/'reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
METADATA=json.loads((HERE/'metadata.json').read_text())


@pytest.fixture(scope='module')
def evaluated():
    records=json.loads((HERE/'data/calibration.json').read_text())
    model=QubitModel();assert model.fit(records) is model
    values=model.predict([r['experiment'] for r in records])
    assert isinstance(values,np.ndarray) and values.shape==(len(records),)
    assert np.isfinite(values).all()
    residual=(values-np.array([r['probability'] for r in records]))/np.array([r['sigma'] for r in records])
    result=dict(parameters={'gamma':model.gamma},parameter_relative_error=abs(model.gamma/reference.TRUE_PARAMETER-1),
                calibration_chi2=float(residual@residual)/(len(records)-1),hidden={})
    for name,experiments in reference.hidden_inputs().items():
        truth=reference.predict(experiments,reference.TRUE_PARAMETER)
        actual=model.predict(experiments)
        assert isinstance(actual,np.ndarray) and actual.shape==truth.shape
        assert np.isfinite(actual).all() and np.all(actual>=-1e-10) and np.all(actual<=1+1e-10)
        result['hidden'][name]=float(np.sqrt(np.mean((actual-truth)**2)))
    path=Path(os.environ.get('QUBIT_METRICS_PATH','/logs/verifier/metrics.json'))
    if path.parent.exists():path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def test_private_calibration(evaluated):
    assert evaluated['calibration_chi2']<1.5


def test_parameter_recovery(evaluated):
    assert evaluated['parameter_relative_error']<METADATA['parameter_limit']


@pytest.mark.parametrize('case',list(reference.hidden_inputs()))
def test_prediction(evaluated,case):
    assert evaluated['hidden'][case]<METADATA['prediction_limit']
