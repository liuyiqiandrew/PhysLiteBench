"""Independent fixed-drag checks; run only after manual submitted-source review."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
STAGE=Path(__file__).resolve().parents[1]
REPO=STAGE.parents[1]
JOB=REPO/'jobs/rotating-reservoir-r7-plain-20261006-042142'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert (JOB/'run-timing.json').exists(),'All three trials must finish'
collected=json.loads((STAGE/'results/rotating-reservoir-r7-results.json').read_text())
reference=load(STAGE/'tasks/rotating-reservoir/tests/reference.py','independent_rotor_r7')
rng=np.random.default_rng(810617)
cases=[]
for i in range(24):
    cases.append(dict(temperature_a=float(rng.uniform(.6,1.6)),temperature_b=float(rng.uniform(.6,1.4)),
        angular_speed=float(rng.uniform(-.65,.65)),flip_rate=float(rng.uniform(.8,2.)),
        flux_fraction=float(rng.uniform(0,1)),heat_bias=float(rng.uniform(-.35,.35)),statistic='scgf'))
for i in range(6):cases.append(dict(cases[i],heat_bias=0.,statistic='mean' if i%2 else 'noise'))
reviews={}
for trial in collected['trials']:
    name=trial['trial'];folder=JOB/name;path=folder/'artifacts/app/model.py'
    if not path.exists() or not trial.get('metrics'):
        reviews[name]=dict(classification='pending_root_source_review',reward=trial['reward'],exception=trial['exception'],reason='Missing final source or verifier metrics; source/native audit required')
        continue
    value=trial['metrics']['parameter'];module=load(path,name);model=module.Model();model.drag=value
    actual=model.predict(cases);expected=reference.predict(cases,value)
    hidden={}
    for group,experiments in reference.hidden_inputs().items():
        a=model.predict(experiments);b=reference.predict(experiments,value)
        hidden[group]=float(np.linalg.norm(a-b)/np.linalg.norm(b))
    trajectory=json.loads((folder/'agent/trajectory.json').read_text())
    messages=[step['message'] for step in trajectory['steps'] if step.get('source') not in ['system','user'] and step.get('message')]
    reviews[name]=dict(reward=trial['reward'],exception=trial['exception'],classification='pending_root_actual_source_review',
        native_drag=value,offgrid_maximum_error=float(np.max(abs(actual-expected))),offgrid_prediction=actual.tolist(),
        offgrid_reference=expected.tolist(),same_parameter_hidden_relative_errors=hidden,
        trajectory_evidence=messages,sha256={str(p.relative_to(folder)):sha(p) for p in [path,folder/'agent/trajectory.json',folder/'result.json']})
    print(name,reviews[name]['offgrid_maximum_error'],hidden,flush=True)
report=dict(task='rotating-reservoir',revision=7,job=JOB.name,reviewer='root',passes=collected['passes'],trial_count=3,
    clean_physical_failures=0,trials=reviews,random_seed=810617,offgrid_inputs=cases,
    scope='Fixed native parameter comparisons and retained trajectory evidence; classification remains pending actual-source causal review. All original submissions, fits and rewards preserved.')
(STAGE/'results/rotating-reservoir-r7-trial-review.json').write_text(json.dumps(report,indent=2)+'\n')
