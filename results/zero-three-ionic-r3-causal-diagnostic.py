"""Fixed-D causal field repair; does not modify submitted or frozen files."""
import hashlib
import importlib.util
import json
from pathlib import Path
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT/'jobs/ionic-current-loops-zero-three-ionic-r3-plain-20261004-074536'
TASK = ROOT/'staging/ionic-current-loops-r3/tasks/ionic-current-loops'

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def nrms(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))

reference=load('ionic_reference',TASK/'tests/reference.py')
records=json.loads((TASK/'environment/data/calibration.json').read_text())
cal=[r['input'] for r in records]
truth={key:reference.predict(inputs) for key,inputs in reference.hidden_inputs().items()}
report={}
for trial in sorted(JOB.glob('ionic-current-loops__*')):
    if not (trial/'result.json').exists():continue
    result=json.loads((trial/'result.json').read_text())
    if result['verifier_result']['rewards']['reward']!=0:continue
    path=trial/'artifacts/app/model.py';original_hash=sha(path);source=path.read_text()
    line="    electric += np.einsum('i,ijab->jab', offset, np.array(responses))"
    assert source.count(line)==1
    patched=source.replace(line,'    # Author diagnostic: retain the purely periodic electrostatic field.')
    original=load('ionic_submitted_'+trial.name,path)
    repaired=types.ModuleType('ionic_repaired_'+trial.name)
    exec(compile(patched,'<field-only-repair>','exec'),repaired.__dict__)
    metrics=json.loads((trial/'verifier/metrics.json').read_text())
    D=metrics['parameter'];before=original.Model();after=repaired.Model()
    before.diffusivity=after.diffusivity=D
    hidden={}
    for key,inputs in reference.hidden_inputs().items():
        raw=before.predict(inputs);fixed=after.predict(inputs)
        hidden[key]={'submitted_nrms':nrms(raw,truth[key]),'repaired_nrms':nrms(fixed,truth[key]),'recorded_metric_difference':abs(nrms(raw,truth[key])-metrics['hidden'][key]),'passes_unchanged_gate':nrms(fixed,truth[key])<.04}
    e=reference.hidden_inputs()['phase_sweep'][4]
    xf,yf,_,Ebefore,fluxbefore,_=original.fields(tuple(e['amplitudes']),e['phase'])
    _,_,_,Eafter,fluxafter,_=repaired.fields(tuple(e['amplitudes']),e['phase'])
    z=np.array([1.,1.,-1.])
    report[trial.name]={'diffusivity_held_fixed':D,'refit':False,'repair':'Remove only the additive zero-net-current response from the electric field; retain every concentration, local potential solve, gradient, flux, readout and fitted parameter.','removed_line':line,'calibration_prediction_max_abs_change':float(np.max(abs(before.predict(cal)-after.predict(cal)))),'hidden':hidden,'example':{'input':e,'submitted_mean_electric_field':Ebefore.mean(axis=(1,2)).tolist(),'repaired_mean_electric_field':Eafter.mean(axis=(1,2)).tolist(),'submitted_mean_charge_current':(np.einsum('i,ijab->jab',z,fluxbefore)*D).mean(axis=(1,2)).tolist(),'repaired_mean_charge_current':(np.einsum('i,ijab->jab',z,fluxafter)*D).mean(axis=(1,2)).tolist()},'source_sha256':original_hash,'original_artifacts_unchanged':sha(path)==original_hash}
out={'scope':'Author-only deterministic causal checks, not model-agent runs.','reference_sha256':sha(TASK/'tests/reference.py'),'script_sha256':sha(Path(__file__)),'trials':report}
(ROOT/'results/zero-three-ionic-r3-causal-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
