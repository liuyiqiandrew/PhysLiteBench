import importlib.util,json,math,os
from pathlib import Path
import numpy as np
import pytest
import reference

HERE=Path(__file__).parent
METADATA=json.loads((HERE/'metadata.json').read_text())
# reference.predict(hidden_inputs, TRUE_PARAMETER), frozen by scripts/freeze_hidden_truth.py so the verifier does not recompute it.
FROZEN=json.loads((HERE/'data/hidden_truth.json').read_text())['groups']
def canonical(value):return json.loads(json.dumps(value,default=lambda x:x.tolist() if hasattr(x,'tolist') else x))
def same_inputs(a,b):
    # Equal structure and strings; numbers equal up to last-bit differences between CPUs.
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same_inputs(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(map(same_inputs,a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-300)
    return a==b

@pytest.fixture(scope='module')
def evaluated():
    path=Path(os.environ.get('MODEL_PATH','/app/model.py'))
    spec=importlib.util.spec_from_file_location('submission',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=json.loads((HERE/'data/calibration.json').read_text());model=module.Model();returned=model.fit(records)
    values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);pred=model.predict([r['input'] for r in records])
    assert isinstance(pred,np.ndarray) and pred.shape==(len(records),) and np.isfinite(pred).all()
    hidden={}
    for key,inputs in reference.hidden_inputs().items():
        assert same_inputs(FROZEN[key]['inputs'],canonical(inputs)),key
        truth=np.array(FROZEN[key]['truth']);got=model.predict(inputs)
        assert isinstance(got,np.ndarray) and got.shape==(len(inputs),) and np.isfinite(got).all()
        hidden[key]=float(np.sqrt(np.mean((got-truth)**2))/np.sqrt(np.mean(truth**2)))
    metrics=dict(parameter_relative_error=abs(model.drag/reference.TRUE_PARAMETER-1),calibration_chi2=float(np.sum(((pred-values)/sigma)**2)/(len(records)-1)),hidden=hidden,parameter=float(model.drag))
    target=Path(os.environ.get('METRICS_PATH','/logs/verifier/metrics.json'));target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(metrics,indent=2))
    return returned is model,pred,metrics

def test_interface(evaluated):assert evaluated[0] and isinstance(evaluated[1],np.ndarray)
def test_finite(evaluated):assert np.isfinite(evaluated[1]).all() and all(np.isfinite(v) for v in evaluated[2]['hidden'].values())
def test_calibration(evaluated):assert evaluated[2]['calibration_chi2']<1.5
def test_parameter(evaluated):assert evaluated[2]['parameter_relative_error']<.03
@pytest.mark.parametrize('case',list(reference.hidden_inputs()))
def test_prediction(evaluated,case):assert evaluated[2]['hidden'][case]<.04
