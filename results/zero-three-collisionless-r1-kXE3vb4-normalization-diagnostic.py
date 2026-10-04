"""In-memory factor-four derivative diagnostic; no submission/score changes."""
import datetime,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'jobs/collisionless-screening-zero-three-collisionless-r1-outage-recovery-20261004-1410/collisionless-screening__kXE3vb4'
REF=ROOT/'staging/collisionless-screening-r1/tasks/collisionless-screening/tests/reference.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
source=TRIAL/'artifacts/app/model.py';before_hash=sha(source);m=module('saved_model',source);ref=module('independent_reference',REF)
metrics=json.loads((TRIAL/'verifier/metrics.json').read_text());density=metrics['parameter']
records=json.loads((TRIAL/'artifacts/app/data/calibration.json').read_text());xs=[x['input'] for x in records]
def repaired(es):
 h,c=m._response_components(es)
 return np.real(1/(1-density*(h.real+.25j*h.imag)/c))
a=m.predict_at(xs,density);b=repaired(xs)
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'trial':TRIAL.name,'classification':'mathematical_normalization_failure','desired_physical_failure':False,'diagnostic_only_no_score_change':True,'parameter_held_fixed':density,'change':'Multiply only the submitted support-masked imaginary response by 1/4, correcting d[15/16*(1-u^2)^2]/du = (15/4)*u*(u^2-1). Causal prescription, support mask, real response and density stay fixed.','calibration_max_prediction_change':float(np.max(abs(a-b))),'groups':{}}
for name,es in ref.hidden_inputs().items():
 truth=ref.predict(es,ref.TRUE_PARAMETER);a=m.predict_at(es,density);b=repaired(es)
 report['groups'][name]={'before_error':float(np.linalg.norm(a-truth)/np.linalg.norm(truth)),'after_error':float(np.linalg.norm(b-truth)/np.linalg.norm(truth)),'before':a.tolist(),'after':b.tolist(),'truth':truth.tolist()}
assert report['calibration_max_prediction_change']==0
assert all(x['after_error']<.04 for x in report['groups'].values())
assert sha(source)==before_hash
report['original_submission_unchanged_sha256']=before_hash
report['evidence_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [source,TRIAL/'result.json',TRIAL/'verifier/metrics.json',TRIAL/'agent/codex.txt',REF,Path(__file__)]}
f=Path(__file__).with_suffix('.json');f.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'path':str(f.relative_to(ROOT)),'sha256':sha(f),'calibration_change':report['calibration_max_prediction_change'],'errors':{k:{kk:vv for kk,vv in v.items() if kk.endswith('error')} for k,v in report['groups'].items()}},indent=2))
