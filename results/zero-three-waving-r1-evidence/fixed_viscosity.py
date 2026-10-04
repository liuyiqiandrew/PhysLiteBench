from pathlib import Path
from types import ModuleType
import hashlib,json,numpy as np
ROOT=Path('/Users/andrewliu/Library/CloudStorage/OneDrive-PrincetonUniversity/Courses/AI Agent/project1/PhysLiteBench')
JOB=ROOT/'jobs/waving-sheet-zero-three-waving-r1-outage-recovery-20261004-1410'
TASK=ROOT/'staging/waving-sheet-r1/tasks/waving-sheet'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,name):
 m=ModuleType(name);exec(compile(p.read_text(),str(p),'exec'),m.__dict__);return m
ref=load(TASK/'tests/reference.py','waving_ref')
records=json.loads((TASK/'tests/data/calibration.json').read_text())
cal=[r['input'] for r in records]
groups=ref.hidden_inputs()
truth={k:ref.predict(v,ref.TRUE_PARAMETER) for k,v in groups.items()}
report={'task':'waving-sheet','revision':1,'kind':'fixed_viscosity_response_only_causal_diagnostic','score_changes':False,'model_evaluations':0,'original_files_mutated':False,'fit_repeated':False,'script_sha256':digest(Path(__file__)),'reference_sha256':digest(TASK/'tests/reference.py'),'repair':'Keep submitted viscosity and exact first-order coefficients. For pumping only add integral_0^infinity <u1*v1> dy / viscosity to the submitted wall-Taylor mean velocity, from stationary mean Navier-Stokes momentum. All other outputs and the submitted fit are unchanged.','trials':{}}
for trial in sorted(JOB.glob('waving-sheet__*')):
 rp=trial/'result.json'
 if not rp.exists():continue
 raw=json.loads(rp.read_text())
 if raw.get('verifier_result',{}).get('rewards',{}).get('reward')!=0 or raw.get('exception_info') is not None:continue
 source=trial/'artifacts/app/model.py';beforehash=digest(source)
 model=load(source,trial.name)
 metrics=json.loads((trial/'verifier/metrics.json').read_text());nu=metrics['parameters']['viscosity']
 reading=model.reading
 def corrected(e,viscosity):
  value=reading(e,viscosity)
  if e['observable']!='pumping':return value
  k,w=e['wave_number'],e['frequency'];p,c=model.coefficients(viscosity,k,w)
  u,v=-p*c,-1j*k*c
  transport=.5*np.real(np.sum(u[:,None]*v.conj()[None,:]/(p[:,None]+p.conj()[None,:])))
  return float(value+transport/viscosity)
 entry={'source_path':str(source.relative_to(ROOT)),'source_sha256':beforehash,'fixed_viscosity':nu,'recorded_metrics':metrics,'before':{},'after':{},'calibration_max_change':float(max(abs(corrected(e,nu)-reading(e,nu)) for e in cal))}
 for key,inputs in groups.items():
  b=np.array([reading(e,nu) for e in inputs]);a=np.array([corrected(e,nu) for e in inputs]);y=truth[key]
  norm=lambda x:float(np.sqrt(np.mean((x-y)**2)/np.mean(y*y)))
  entry['before'][key]=norm(b);entry['after'][key]=norm(a)
 entry['all_repaired_groups_pass_unchanged_gate']=max(entry['after'].values())<.04
 entry['original_hash_unchanged']=digest(source)==beforehash
 assert entry['original_hash_unchanged'] and entry['all_repaired_groups_pass_unchanged_gate']
 report['trials'][trial.name]=entry
out=ROOT/'results/zero-three-waving-r1-fixed-viscosity-diagnostic.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'path':str(out.relative_to(ROOT)),'sha256':digest(out),'trials':report['trials']},indent=2))
