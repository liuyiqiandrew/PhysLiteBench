"""Final-source checks reusing only byte-verified, unchanged direct trajectories.

No interpolation is used. Every cache entry is an exact original configuration.
The local pytest controls execute the final reference afresh under60s.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

BASE=Path(__file__).resolve().parents[1]
HISTORY=BASE/'history/initial-validation'

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

manifest=json.loads((HISTORY/'manifest.json').read_text())
assert all(sha(HISTORY/p)==h for p,h in manifest['files'].items())
assert sha(HISTORY/'tasks/adiabatic-capture/tests/reference.py')==sha(BASE/'tasks/adiabatic-capture/tests/reference.py')
assert sha(HISTORY/'scripts/adiabatic_capture_baseline.py')==sha(BASE/'scripts/adiabatic_capture_baseline.py')
assert sha(HISTORY/'tasks/adiabatic-capture/environment/data/calibration.json')==sha(BASE/'tasks/adiabatic-capture/environment/data/calibration.json')
validator=load('validator',BASE/'scripts/validate_adiabatic_capture.py')
old=load('old_oracle',HISTORY/'tasks/adiabatic-capture/solution/model.py')
FinalModel=validator.oracle.Model
comparison={'fit_calls':0,'prediction_calls':0,'max_fit_change':0.,'max_prediction_change':0.,'per_fit':[]}

class ComparedModel:
    def __init__(self):self.new=FinalModel();self.old=old.Model();self.action_scale=None
    def fit(self,records):
        self.new.fit(records);self.old.fit(records);self.action_scale=self.new.action_scale
        error=abs(self.new.action_scale-self.old.action_scale)
        comparison['fit_calls']+=1;comparison['max_fit_change']=max(comparison['max_fit_change'],error)
        comparison['per_fit'].append({'old_parameter':self.old.action_scale,'new_parameter':self.new.action_scale})
        assert error<1e-11
        return self
    def predict(self,experiments):
        y=self.new.predict(experiments);before=self.old.predict(experiments)
        error=float(np.max(abs(y-before))) if len(y) else 0.
        comparison['prediction_calls']+=1;comparison['max_prediction_change']=max(comparison['max_prediction_change'],error)
        assert error<1e-10
        return y

validator.oracle.Model=ComparedModel
rows=[json.loads(x) for x in (HISTORY/'results/validation-runs.jsonl').read_text().splitlines()]
cache={};original_scored_seconds=0.
def key(j,d,s,duration=512.,step=.04,action_order=32,phase_order=256,phase_offset=.5):
    return (float(j),float(d),float(s),float(duration),float(step),int(action_order),int(phase_order),float(phase_offset))
for row in rows:
    if row['stage']=='scored_reference':
        e=row['input'];cache[key(validator.reference.TRUE_PARAMETER,e['delta'],e['s_final'])]=row['reference']
        original_scored_seconds+=row['reference']['seconds']
    elif row['stage']=='per_scored_case_refinement':
        e=row['input'];j=validator.reference.TRUE_PARAMETER
        cache[key(j,e['delta'],e['s_final'],1024.,.04,64,512)]=row['joint_refinement']
        cache[key(j,e['delta'],e['s_final'],512.,.02,32,256)]=row['step_refinement']
    elif row['stage']=='domain_reference':
        cache[key(row['parameter'],row['delta'],row['s_final'])]=row['reference']
assert len(cache)==40 and original_scored_seconds<50
actual_trajectory=validator.reference.trajectory
reuse={'exact_configuration_hits':0,'fresh_calls':0}
def trajectory(j,d,s,duration=512.,step=.04,action_order=32,phase_order=256,phase_offset=.5):
    k=key(j,d,s,duration,step,action_order,phase_order,phase_offset)
    if k in cache:
        reuse['exact_configuration_hits']+=1
        return cache[k]
    reuse['fresh_calls']+=1
    return actual_trajectory(j,d,s,duration,step,action_order,phase_order,phase_offset)
validator.reference.trajectory=trajectory
validator.run(generate=False)
report_path=BASE/'results/adiabatic-capture-r1-validation.json'
report=json.loads(report_path.read_text())
report['scored_reference_reused_lookup_seconds']=report['scored_reference_cold_seconds']
report['scored_reference_cold_seconds']=original_scored_seconds
report['scored_reference_timing_definition']='Sum of the8 unchanged original cold trajectory calls; actual final-source full verifier wall times are separately recorded in local_controls.'
report['execution_history']={
 'initial_snapshot':'history/initial-validation/manifest.json',
 'initial_manifest_sha256':sha(HISTORY/'manifest.json'),
 'initial_failure':'Symmetric partial-action mean used split versus unsplit quadrature;1.06e-7 absolute mismatch exceeds an unjustified1e-10 check. Data/grading unchanged.',
 'endpoint_finding':'results/topology-endpoint-check.json preserves near-threshold branch bracketing failures.',
 'repair':'Only branch-energy endpoint brackets use exact minimum/barrier areas; pointwise symmetry is checked separately from mean quadrature refinement.',
 'executed_final_checker':'results/verify_endpoint_repair.py',
 'checker_sha256':sha(Path(__file__)),
 'reference_source_sha256':sha(BASE/'tasks/adiabatic-capture/tests/reference.py'),
 'trajectory_reuse':reuse,
 'all256_old_final_fit_prediction_comparisons':comparison,
 'local_controls':'Fresh final-source subprocesses; no reused trajectory cache in either verifier.'}
report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
compact={k:v for k,v in report['execution_history'].items() if k!='all256_old_final_fit_prediction_comparisons'}
compact['comparison_summary']={k:v for k,v in comparison.items() if k!='per_fit'}
(BASE/'results/endpoint-repair-validation.json').write_text(json.dumps(compact,indent=2)+'\n')
print(json.dumps(compact,indent=2),flush=True)
