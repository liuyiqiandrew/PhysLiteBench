"""Posthoc in-memory repairs at the submitted density; no task or trial edits."""
from pathlib import Path
import importlib.util
import hashlib
import json
import types
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT/'jobs/pressure-surface-waves-zero-three-pressurewaves-r1-plain-20261004-070231'
TRIAL = JOB/'pressure-surface-waves__8kim2xP'
SOURCE = TRIAL/'artifacts/app/model.py'
REFERENCE = JOB/'frozen-task/tests/reference.py'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def module_from_text(name, text):
    m=types.ModuleType(name);exec(compile(text,name,'exec'),m.__dict__);return m
original=module_from_text('original_submission',SOURCE.read_text())
text=SOURCE.read_text()
needle='    return np.array([shear_traction, normal_traction])'
assert text.count(needle)==1
replacement='    shear_traction -= (c-b)*normal\n    normal_traction -= (c-b)*tangent\n'+needle
repaired_text=text.replace(needle,replacement)
repaired=module_from_text('pressure_traction_repair',repaired_text)
spec=importlib.util.spec_from_file_location('reference',REFERENCE)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
metrics=json.loads((TRIAL/'verifier/metrics.json').read_text())
rho=metrics['parameters']['density']
records=json.loads((JOB/'frozen-task/tests/data/calibration.json').read_text())
inputs=[r['input'] for r in records]
values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
def chi(p): return float(np.sum(((p-values)/sigma)**2)/(len(records)-1))
def error(pred,truth): return float(np.sqrt(np.mean((pred-truth)**2)/np.mean(truth**2)))

def probe_mode(e,correct_ratio):
    s,t=original.geometry(e);b=t*t;c=1-2*np.log(s*t);D=b+c+2
    def f(x):
        ql=np.sqrt(1-b*x/D);qt=np.sqrt(1-x)
        factor=1. if correct_ratio else (c+2)/D
        hL=factor*ql;hT=factor/qt
        return np.linalg.det([[-b*(ql+hL),-b*(qt+hT)],
                              [(c+2-b)-D*ql*hL,(c+2-b)-D*qt*hT]])
    # The agent scanned a 10001-point interval; use the identical interval and
    # isolate its nontrivial root. Finer scan changes no equation or parameter.
    xs=np.linspace(1e-8,1-1e-8,10001);ys=np.array([f(x) for x in xs])
    brackets=np.flatnonzero(ys[:-1]*ys[1:]<0)
    assert len(brackets)==1,(e,brackets)
    i=brackets[0];x=brentq(f,xs[i],xs[i+1],xtol=1e-13)
    return s*s-b+b*x
cache={}
def probe_predict(experiments,correct_ratio):
    answer=[]
    for e in experiments:
        key=(e['stretch'],e['pressure'],correct_ratio)
        if key not in cache:cache[key]=probe_mode(e,correct_ratio)
        answer.append(np.sqrt(original.MU*cache[key]/rho))
    return np.array(answer)

report={
 'task':'pressure-surface-waves','revision':1,'trial':TRIAL.name,
 'diagnostic_only':True,'new_agent_evaluations':0,'original_artifacts_modified':False,
 'fixed_parameter':{'density':rho},
 'evidence':{str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,TRIAL/'agent/trajectory.json',TRIAL/'verifier/metrics.json',REFERENCE,Path(__file__)]},
 'final_source_repair':{'change':'In surface_matrix only, subtract (c-b)*normal from shear_traction and (c-b)*tangent from normal_traction. Static equilibrium gives c-b=pressure*J/MU. No density refit or other code change.', 'repaired_source_sha256':hashlib.sha256(repaired_text.encode()).hexdigest()},
 'exploratory_probe':{'trajectory_step':8,'original_ratios':'hL=(c+2)*ql/D; hT=(c+2)/(D*qt)', 'algebra_only_repair':'hL=ql; hT=1/qt', 'derivation':'For the displacement columns (i,-h), the bulk acoustic equation gives h=(s²+c+2-b*q²-y)/((c+2)*q), with y=s²-b+b*x. For ql²=1-b*x/D it gives h=ql; for qt²=1-x it gives h=1/qt. The agent matrix already contains follower-pressure coefficients b and c+2-b, using pJ=c-b.', 'calibration_chi2':{}},
 'hidden':{},
}
for name,m in [('original',original),('pressure_traction_repair',repaired)]:
 report.setdefault('calibration_chi2',{})[name]=chi(m.predict_at(inputs,rho))
for correct,label in [(False,'original_probe'),(True,'ratio_repaired_probe')]:
 report['exploratory_probe']['calibration_chi2'][label]=chi(probe_predict(inputs,correct))
for name,experiments in reference.hidden_inputs().items():
 truth=reference.predict(experiments,reference.TRUE_PARAMETER)
 report['hidden'][name]={
 'original':error(original.predict_at(experiments,rho),truth),
 'pressure_traction_repair':error(repaired.predict_at(experiments,rho),truth),
 'original_probe':error(probe_predict(experiments,False),truth),
 'ratio_repaired_probe':error(probe_predict(experiments,True),truth),
 'repaired_probe_vs_repaired_source_max_absolute':float(np.max(abs(probe_predict(experiments,True)-repaired.predict_at(experiments,rho))))}
report['classification']='implementation_induced_regression'
report['qualification']='The final model uses the wrong physical boundary closure, but its preceding probe already encodes follower-pressure traction. An attenuation-eigenvector algebra error makes that probe fail even at zero pressure; the next public update retains the supplied closure because calibration matches it. The algebra-only repair recovers calibration and hidden predictions at unchanged density. Conservatively exclude this as a clean target physical failure; the exact internal reason for retreat is not observable.'
assert sha(SOURCE)==report['evidence'][str(SOURCE.relative_to(ROOT))]
out=ROOT/'results/zero-three-pressurewaves-r1-causal-diagnostic.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
