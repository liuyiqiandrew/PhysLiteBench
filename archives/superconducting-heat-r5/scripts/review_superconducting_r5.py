"""Review immutable r5 submissions against independent Fock/spin calorimetry."""
from pathlib import Path
import importlib.util,hashlib,json
import numpy as np
STAGE=Path(__file__).resolve().parents[1]
REPO=STAGE.parents[1]
JOB=REPO/'jobs/superconducting-heat-r5-plain-20261006-035908'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert (JOB/'run-timing.json').exists(),'Batch must finish before final review'
collected=json.loads((STAGE/'results/superconducting-heat-r5-results.json').read_text())
reference=load(STAGE/'tasks/superconducting-heat/tests/reference.py','independent_r5_fock')
rng=np.random.default_rng(716105);cases=[]
for i in range(24):
    cases.append(dict(left_temperature=float(rng.uniform(.25,1.05)),right_temperature=float(rng.uniform(.25,1.05)),
      phase=float(rng.uniform(-np.pi,np.pi)),amplitude=float(rng.uniform(0,1)),
      drive_energy=float(rng.uniform(.1,.5) if i%2 else rng.uniform(7,12)),
      oscillator_coupling=float(rng.uniform(0,1)),oscillator_temperature=float(rng.uniform(12,24)),
      statistic='phase_contrast' if i%2 else 'mean'))
reviews={}
for trial in collected['trials']:
    name=trial['trial'];folder=JOB/name;model_path=folder/'artifacts/app/model.py'
    if not model_path.exists():
        reviews[name]=dict(classification='missing_submission',reward=trial['reward'],exception=trial['exception']);continue
    module=load(model_path,name)
    source=model_path.read_text()
    value=trial['metrics']['parameters']['conductance']
    model=module.Model();model.conductance=value
    actual=model.predict(cases);expected=reference.predict(cases,value)
    hidden={}
    for group,inputs in reference.hidden_inputs().items():
        a=model.predict(inputs);b=reference.predict(inputs,value)
        hidden[group]=float(np.linalg.norm(a-b)/np.linalg.norm(b))
    trajectory=json.loads((folder/'agent/trajectory.json').read_text())
    messages=[step['message'] for step in trajectory['steps'] if step.get('source') not in ['system','user'] and step.get('message')]
    reviews[name]=dict(reward=trial['reward'],exception=trial['exception'],
      classification='pending_root_actual_source_review',native_conductance=value,
      uses_supplied_exact_branch=('FACTORIZED = False' in source),
      independent_maximum_error_pW=float(np.max(abs(actual-expected))),
      same_parameter_hidden_relative_errors=hidden,trajectory_evidence=messages,
      sha256={str(p.relative_to(folder)):sha(p) for p in [model_path,folder/'agent/trajectory.json',folder/'result.json']})
    print(name,reviews[name]['independent_maximum_error_pW'],hidden,flush=True)
report=dict(task='superconducting-heat',revision=5,job=JOB.name,reviewer='root',
    trials=reviews,passes=collected['passes'],trial_count=3,clean_physical_failures=0,
    random_seed=716105,offgrid_inputs=cases,
    public_information_issue='The public scaffold accidentally included an unused exact sideband-coherence branch. Every outcome is preserved; revision6 removing it was predeclared before any outcomes. This batch cannot qualify the corrected public package.',
    source_public_correction='results/superconducting-heat-r5-public-source-review-addendum.json')
path=STAGE/'results/superconducting-heat-r5-trial-reviews.json'
path.write_text(json.dumps(report,indent=2)+'\n')
