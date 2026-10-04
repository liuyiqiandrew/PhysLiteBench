"""Posthoc fixed-parameter diagnostic; does not modify any retained trial artifact."""
from pathlib import Path
import hashlib,importlib.util,json,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'jobs/spin-strain-response-neutral-r1-initial-plain-20261003-053210/spin-strain-response__u22cuVy'
TASK=ROOT/'tasks/spin-strain-response'
SOURCE=TRIAL/'artifacts/app/model.py'
original=SOURCE.read_text();before_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
old='''    q2_mean = np.dot(weights, np.diag(q_in_energy_basis @ q_in_energy_basis))
    return float((q2_mean - q_mean * q_mean) / temperature)'''
new='''    result = (np.dot(weights, np.diag(q_in_energy_basis)**2) - q_mean*q_mean)/temperature
    for i in range(3):
        for j in range(i+1,3):
            gap = energy[j] - energy[i]
            quotient = weights[i]/temperature if gap == 0. else weights[i]*(-np.expm1(-gap/temperature))/gap
            result += 2.*quotient*q_in_energy_basis[i,j]**2
    return float(result)'''
assert original.count(old)==1
corrected=original.replace(old,new)
submitted=types.ModuleType('submitted');exec(compile(original,str(SOURCE),'exec'),submitted.__dict__)
repaired=types.ModuleType('repaired');exec(compile(corrected,'in_memory_physical_repair','exec'),repaired.__dict__)
spec=importlib.util.spec_from_file_location('reference',TASK/'tests/reference.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
metrics=json.loads((TRIAL/'verifier/metrics.json').read_text());coupling=metrics['parameters']['coupling']
records=json.loads((TASK/'tests/data/calibration.json').read_text());cal=[r['input'] for r in records];observed=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);groups=reference.hidden_inputs()
report={'trial':TRIAL.name,'classification':'physics_model_failure','fixed_parameter':{'name':'coupling','value':coupling},'change':'Replace only the equal-time variance response by the equilibrium Gibbs derivative in the submitted response function. Keep Hamiltonian, exact diagonalization, thermal weights, fit code, parameter, API and calibration unchanged.','before_source_block':old,'after_source_block':new,'results':{}}
cal_predictions={}
for name,module in [('submitted',submitted),('physical_repair',repaired)]:
 model=module.Model();model.coupling=coupling
 predicted=model.predict(cal);cal_predictions[name]=predicted;residual=(predicted-observed)/sigma
 hidden={}
 for k,experiments in groups.items():
  truth=reference.predict(experiments);actual=model.predict(experiments)
  hidden[k]=float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))
 report['results'][name]={'calibration_chi2':float(residual@residual/(len(cal)-1)),'hidden':hidden}
assert max(abs(report['results']['submitted']['hidden'][k]-metrics['hidden'][k]) for k in groups)<1e-12
assert all(v<.04 for v in report['results']['physical_repair']['hidden'].values())
report['calibration_prediction_max_change']=float(np.max(abs(cal_predictions['submitted']-cal_predictions['physical_repair'])))
assert report['calibration_prediction_max_change']<1e-12
report['all_hidden_groups_pass_after_repair']=True
report['submitted_source_sha256']=before_hash
report['retained_artifact_unchanged']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before_hash
assert report['retained_artifact_unchanged']
report['execution']='In-memory source replacement, fixed submitted coupling, no refit and no retained artifact edits.'
report['diagnostic_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
p=ROOT/'results/spin-strain-response-initial-repair.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
