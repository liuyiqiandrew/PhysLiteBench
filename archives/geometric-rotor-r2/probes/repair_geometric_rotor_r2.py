from pathlib import Path
import json,hashlib,importlib.util,re,ast
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
root=Path.cwd();trial=root/'jobs/geometric-rotor-neutral-r2-initial-plain-20261003-045716/geometric-rotor__8H4GDTH';source=Path('/private/tmp/rotor_r2_trial_diagnostic_original.py').read_text()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
agent=load('agent',trial/'artifacts/app/model.py');oracle=load('oracle',root/'tasks/geometric-rotor/solution/model.py');reference=load('reference',root/'tasks/geometric-rotor/tests/reference.py')
records=json.loads((root/'tasks/geometric-rotor/tests/data/calibration.json').read_text());inputs=[r['input'] for r in records];values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);I=json.loads((trial/'verifier/metrics.json').read_text())['parameters']['inertia']
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='lev');definition=ast.get_source_segment(source,node);scope={'np':np,'eigh':eigh,'Ns':256,'M':10};exec(definition,scope);lev=scope['lev'];cache={}
def predict(experiments,factor,cols=slice(0,2)):
 out=[]
 for e in experiments:
  key=(e['theta'],e['q'],e['r'],cols.start,cols.stop)
  if key not in cache:cache[key]=lev(e['theta'],e['q'],e['r'],cols)
  energy=factor*cache[key]/I;weight=np.exp(-(energy-energy.min())/e['temperature']);out.append(weight@energy/weight.sum())
 return np.array(out)
def chi(y):return float(np.sum(((y-values)/sigma)**2)/(len(values)-1))
original=predict(inputs,1.);corrected=predict(inputs,.5);submitted=agent.predict_at(inputs,I);correct=oracle.predict_at(inputs,I)
groups={}
for name,e in reference.hidden_inputs().items():
 truth=reference.predict(e);den=np.sqrt(np.mean(truth**2))
 variants={'submitted':agent.predict_at(e,I),'diagnostic_original':predict(e,1.),'diagnostic_half_factor':predict(e,.5),'physical_oracle_fixed_inertia':oracle.predict_at(e,I)}
 groups[name]={k:float(np.sqrt(np.mean((y-truth)**2))/den) for k,y in variants.items()}
assert chi(original)>1e5 and chi(corrected)<1.5
assert all(g['diagnostic_half_factor']<.04 and g['physical_oracle_fixed_inertia']<.04 for g in groups.values())
report={'task':'geometric-rotor','revision':2,'trial':trial.name,'job':trial.parent.name,'classification':'implementation_failure','failure_subtype':'implementation_induced_model_regression','fixed_fitted_inertia':I,'exact_original_tool_call':'call_vLpEz8BKTjDJ865V8pjwAt03','exact_original_diagnostic':source,'minimal_repair':'The diagnostic builds the complete derivative Gram matrix H, whose physical kinetic operator is H/(2*inertia). Replace return eigh(H,eigvals_only=True) by return eigh(H/2,eigvals_only=True); retain every connection term, endpoint parity, Fourier cutoff, state choice and the fitted inertia.','original_diagnostic_sha256':hashlib.sha256(source.encode()).hexdigest(),'submitted_source_sha256':hashlib.sha256((trial/'artifacts/app/model.py').read_bytes()).hexdigest(),'calibration_chi2':{'diagnostic_original':chi(original),'diagnostic_half_factor':chi(corrected),'submitted':chi(submitted),'physical_oracle_fixed_inertia':chi(correct)},'maximum_corrected_diagnostic_vs_oracle_calibration_difference':float(np.max(abs(corrected-correct))),'maximum_submitted_vs_correct_calibration_difference':float(np.max(abs(submitted-correct))),'maximum_lower_vs_upper_doublet_diagnostic_difference':float(np.max(abs(predict(inputs,.5,slice(0,2))-predict(inputs,.5,slice(2,4))))),'hidden':groups,'causal_evidence':'The public tool trajectory first constructs all derivative terms, corrects an earlier double FFT normalization, and retries both lower/upper columns. The remaining factor-two kinetic scaling causes huge false calibration errors. It then explicitly elects to preserve the supplied calibrated prediction path. Thus the final physical omission follows a demonstrable arithmetic error in an otherwise physically complete diagnostic; it is excluded from the intended physical-failure count.','artifacts_modified':False}
p=root/'results/geometric-rotor-r2-initial-repair.json';p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='exact_original_diagnostic'},indent=2))
