"""Frozen private calibration and joint coincidence checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from model import Model

CALIBRATION_SHA256='8aa1d1a0aa1c001cb01e8f6ce93bfbd076cb1a16bb8b01d2363652fffc977082'
TRUTH_SHA256='6e2d7300a03233d7ee9462a4cbe00cc468ec9e866b5636e3290763303330ca01'
GROUPS=('state_pairs','state_triples','parent_collisions','joint_repeated','joint_mixed','joint_distinct')


@pytest.fixture(scope='session')
def evaluated():
    calibration=Path('/tests/data/calibration.json')
    truth_path=Path('/tests/data/prediction_truth.json')
    assert hashlib.sha256(calibration.read_bytes()).hexdigest()==CALIBRATION_SHA256
    assert hashlib.sha256(truth_path.read_bytes()).hexdigest()==TRUTH_SHA256
    data=json.loads(calibration.read_text());truth=json.loads(truth_path.read_text())
    assert len(data)==192 and len(truth)==58
    model=Model();assert model.fit(data) is model
    parameter=float(model.width)
    predictions=np.asarray(model.predict([r['input'] for r in truth]),dtype=float)
    target=np.asarray([r['value'] for r in truth],dtype=float)
    calibration_predictions=np.asarray(model.predict_calibration(data),dtype=float)
    assert predictions.shape==target.shape==(58,)
    assert calibration_predictions.shape==(192,)
    assert np.all(np.isfinite(predictions)) and np.all(np.isfinite(calibration_predictions))
    errors={}
    for group in GROUPS:
        idx=[i for i,r in enumerate(truth) if group in r['groups']]
        assert idx
        errors[group]=float(np.linalg.norm(predictions[idx]-target[idx])/np.linalg.norm(target[idx]))
    measured=np.asarray([r['value'] for r in data]);sigma=np.asarray([r['sigma'] for r in data])
    metrics={'width':parameter,'parameter_relative_error':abs(parameter/1.03-1),
             'chi2_per_record':float(np.mean(((calibration_predictions-measured)/sigma)**2)),
             'groups':errors,'prediction_values':predictions.tolist(),'reference_values':target.tolist(),
             'actual_fit_count':1}
    folder=Path('/logs/verifier')
    if folder.is_dir():(folder/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    return metrics


@pytest.mark.parametrize('group',GROUPS)
def test_coincidences(evaluated,group):
    assert evaluated['groups'][group]<.04


def test_parameter_and_calibration(evaluated):
    assert evaluated['parameter_relative_error']<.03
    assert evaluated['chi2_per_record']<1.5
