"""Response-only causal diagnostic; does not modify any submission or score."""
import datetime,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'jobs/collisionless-screening-zero-three-collisionless-r1-plain-20261004-134611/collisionless-screening__wjTyWn9'
TASK=ROOT/'staging/collisionless-screening-r1/tasks/collisionless-screening'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
source=TRIAL/'artifacts/app/model.py';source_hash=sha(source)
m=load_module('saved_submission',source);ref=load_module('independent_reference',TASK/'tests/reference.py')
metrics=json.loads((TRIAL/'verifier/metrics.json').read_text());density=metrics['parameter']
records=json.loads((TRIAL/'artifacts/app/data/calibration.json').read_text())
inputs=[x['input'] for x in records];values=np.array([x['value'] for x in records]);sigma=np.array([x['sigma'] for x in records])
def causal(es,masked):
 k=np.array([e['wavenumber'] for e in es]);w=np.array([e['velocity_width'] for e in es]);z=np.array([e['frequency'] for e in es])/(k*w)
 imaginary=15*np.pi/4*(z**3-z)
 if masked:imaginary=imaginary*(np.abs(z)<1)
 return np.real(1/(1-density*(m.velocity_response(z)+1j*imaginary)/(k*w)**2))
original=m.predict_at(inputs,density);corrected=causal(inputs,True);flawed=causal(inputs,False)
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'trial':TRIAL.name,'diagnostic_only_no_score_change':True,'parameter_held_fixed':density,'physical_change':'Complete the causal susceptibility with pi*Fprime(z) only on the support |z|<1, keeping the submitted real PV helper and fitted density.','trajectory_confound':'Public item6 evaluated the imaginary polynomial outside compact support; item7 rejected the causal law on that invalid calibration comparison.','classification_unchanged':'mixed_physics_numerical_failure','desired_physical_failure':False,'calibration':{'original_chi2':float(np.sum(((original-values)/sigma)**2)/(len(records)-1)),'proper_mask_chi2':float(np.sum(((corrected-values)/sigma)**2)/(len(records)-1)),'malformed_unmasked_probe_chi2':float(np.sum(((flawed-values)/sigma)**2)/(len(records)-1)),'proper_mask_max_prediction_change':float(np.max(abs(original-corrected)))},'groups':{}}
for name,es in ref.hidden_inputs().items():
 truth=ref.predict(es,ref.TRUE_PARAMETER);before=m.predict_at(es,density);after=causal(es,True)
 report['groups'][name]={'before_error':float(np.linalg.norm(before-truth)/np.linalg.norm(truth)),'after_error':float(np.linalg.norm(after-truth)/np.linalg.norm(truth)),'truth':truth.tolist(),'before':before.tolist(),'after':after.tolist()}
assert all(x['after_error']<.04 for x in report['groups'].values())
assert report['calibration']['proper_mask_max_prediction_change']<1e-12
assert sha(source)==source_hash
report['original_submission_unchanged_sha256']=source_hash
paths=[source,TRIAL/'result.json',TRIAL/'verifier/metrics.json',TRIAL/'agent/codex.txt',TASK/'tests/reference.py',Path(__file__)]
report['evidence_sha256']={str(p.relative_to(ROOT)):sha(p) for p in paths}
out=ROOT/'results/zero-three-collisionless-r1-fixed-parameter-diagnostic.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'path':str(out.relative_to(ROOT)),'sha256':sha(out),'calibration':report['calibration'],'groups':{k:{kk:vv for kk,vv in v.items() if kk.endswith('error')} for k,v in report['groups'].items()}},indent=2))
