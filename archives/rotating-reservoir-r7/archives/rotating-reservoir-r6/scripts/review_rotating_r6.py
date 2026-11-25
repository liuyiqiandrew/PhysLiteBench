"""Preserve and audit every actual rotor r6 outcome without resampling."""
from pathlib import Path
import importlib.util,json,hashlib,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];NAME='rotating-reservoir-r6-plain-20261006-032901'
JOB=ROOT/'jobs'/NAME
if not JOB.exists():JOB=ROOT.parents[1]/'jobs'/NAME
TASK=JOB/'frozen-task';RESULTS=ROOT/'results'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    start=time.perf_counter();collection=json.loads((RESULTS/'rotating-reservoir-r6-results.json').read_text())
    oracle=load(TASK/'solution/model.py','rotor6_oracle');ref=load(TASK/'tests/reference.py','rotor6_spectral_ref')
    rng=np.random.default_rng(613106);preparations=[]
    for _ in range(32):preparations.append(ref.experiment(rng.uniform(-.65,.65),rng.uniform(.8,2.),rng.uniform(.6,1.6),rng.uniform(.6,1.4),rng.uniform(0,1)))
    for omega,flip,ta,tb,eta in [(0.,.8,1.4,1.4,1.),(.65,.8,.6,1.4,1.),(-.65,2.,1.6,.6,1.),(.65,1.,1.,1.,.5),(-.65,1.,1.4,1.4,0.),(0.,2.,.6,1.4,.5),(0.,.8,1.6,.6,0.),(-.5,1.4,.6,.6,1.)]:preparations.append(ref.experiment(omega,flip,ta,tb,eta))
    experiments=[dict(e,statistic=stat) for e in preparations for stat in ['mean','noise']]
    records=json.loads((TASK/'environment/data/calibration.json').read_text());cal=[r['input'] for r in records]
    rows=[]
    for collected in collection['trials']:
        folder=JOB/collected['trial'];source=folder/'artifacts/app/model.py';assert sha(source)==collected['source_sha256']
        m=load(source,collected['trial']);drag=collected['metrics']['parameter'];model=m.Model();model.drag=drag
        native=next((folder/'agent/sessions').rglob('*.jsonl'));events=[json.loads(line) for line in native.read_text().splitlines()]
        contexts=[row['payload'] for row in events if row.get('type')=='turn_context'];assert contexts and all(c['model']=='gpt-5.6-luna' and c['effort']=='high' for c in contexts)
        messages=[];complete=[];errors=[]
        for i,event in enumerate(events,1):
            payload=event.get('payload',{})
            if event.get('type')=='event_msg' and payload.get('type') in ['agent_message','task_complete']:
                entry=dict(native_line=i,type=payload['type'],message=payload.get('message') or payload.get('last_agent_message'));messages.append(entry)
                if payload['type']=='task_complete':complete.append(entry)
            if event.get('type')=='event_msg' and payload.get('type')=='error':errors.append(dict(native_line=i,payload=payload))
        actual=model.predict(experiments);right=ref.predict(experiments,drag);physical=oracle.predict_at(experiments,drag)
        public={name:sha(folder/'artifacts/app'/name)==sha(TASK/'environment'/name) for name in ['README.md','test_public.py','data/calibration.json']};assert all(public.values())
        row=dict(trial=collected['trial'],reward=collected['reward'],exception_type=(collected['exception'] or {}).get('exception_type'),scientific_outcome_observed=collected['scientific_outcome_observed'],native_completed=bool(complete),native_contexts=collected['native_contexts'],fixed_native_drag=drag,source_sha256=sha(source),native_sha256=sha(native),trajectory_sha256=sha(folder/'agent/trajectory.json'),public_files_match_frozen=public,native_messages=messages,native_error_events=errors,
                 offgrid_independent_max_abs=float(max(abs(actual-right))),offgrid_independent_relative_RMSE=relative(actual,right),per_statistic={stat:dict(max_abs=float(max(abs(actual[i::2]-right[i::2]))),relative_RMSE=relative(actual[i::2],right[i::2])) for i,stat in enumerate(['mean','noise'])},oracle_independent_max_abs=float(max(abs(physical-right))),actual_predictions=actual.tolist(),independent_predictions=right.tolist(),native_hidden_metrics=collected['metrics'])
        if source.parent.parent.parent.name.endswith('FnqWRaG') or collected['trial'].endswith('FnqWRaG'):
            row['physical_assessment']='Correct continuous collision-flux law, joint motor state, relative gas heat and post-jump Poisson correction are implemented. The fixed central finite-volume grid has ordinary discretization error, and replacing a Poisson equation at a vanishing-tail row can amplify floating-point cancellation.'
            text=source.read_text();old='    poisson_system[0, :] = stationary\n    poisson_rhs = mean - event_mean\n    poisson_rhs[0] = 0.0';new='    pivot = int(np.argmax(stationary))\n    poisson_system[pivot, :] = stationary\n    poisson_rhs = mean - event_mean\n    poisson_rhs[pivot] = 0.0';assert text.count(old)==1
            repair=RESULTS/'rotating-reservoir-r6-review-evidence/FnqWRaG-numerical-gauge-repair.py';repair.parent.mkdir(parents=True,exist_ok=True);repair.write_text(text.replace(old,new));fixed=load(repair,'rotor6_gauge_repair');f=fixed.Model();f.drag=drag;values=f.predict(experiments)
            row['numerical_gauge_repair']=dict(path=str(repair.relative_to(ROOT)),sha256=sha(repair),scope='Select maximum stationary probability as the Poisson gauge row, replacing an ill-conditioned tail row. The exact generator, physical law, mesh, fit, drag, heat and full correlation equations are unchanged. This is a numerical repair, not a physical-model correction.',calibration_change_max_abs=float(max(abs(f.predict(cal)-model.predict(cal)))),original_independent_max_abs=float(max(abs(actual-right))),repaired_independent_max_abs=float(max(abs(values-right))),repaired_independent_relative_RMSE=relative(values,right))
            row['classification']='usage_limit_interrupted_actual_reward_pass_with_correct_flux_equations_and_finite_grid_numerics'
        elif collected['trial'].endswith('JF9FXP7'):
            row['physical_assessment']='Correct Gaussian collision marks weighted by the literal relative-speed kernel are assembled in a Hermite-Galerkin generator using analytic absolute Gaussian moments. Both motor labels, relative heat, fourth-mark integrals and complete post-jump Poisson noise are retained. Finite basis and high-order cancellation give numerical approximation error; no Maxwell-clock physical shortcut remains.'
            row['classification']='usage_limit_interrupted_actual_reward_pass_with_correct_flux_Galerkin_and_counting_equations'
        else:
            row['physical_assessment']='Correct collision-flux generator and full Poisson correlations are implemented with refined central velocity grids and exact Maxwell closure at eta0. This is a completed physical solution with ordinary mesh/tail error.'
            row['classification']='completed_reward_pass_with_correct_flux_generator_and_correlated_heat'
            assert complete and row['offgrid_independent_max_abs']<1e-4
        rows.append(row);print('REVIEWED',collected['trial'],row['classification'],row['offgrid_independent_max_abs'],flush=True)
    report=dict(task='rotating-reservoir',revision=6,job=JOB.name,model='gpt-5.6-luna',effort='high',hint=False,reward_passes=3,completed_scientific_trials=1,usage_limit_interrupted_trials=2,qualification='Does not qualify: one complete physical solution passes, and two additional actual installed solutions receive passing verifier rewards despite API usage-limit interruptions. Preserve all raw passes and both infrastructure exceptions; do not call them three completed science trials or reclassify them as failures.',source_matches_frozen_plan=collection['source_matches_frozen_plan'],task_checksum=collection['trials'][0]['task_checksum'],fixed_parameter_seed=613106,random_interior_count=32,physical_anchor_count=8,offgrid_preparations=40,offgrid_readouts=80,offgrid_inputs=experiments,review=rows,seconds=time.perf_counter()-start,
                immutable_frozen_source_sha256={str(path.relative_to(TASK)):sha(path) for path in sorted(TASK.rglob('*')) if path.is_file()})
    path=RESULTS/'rotating-reservoir-r6-trial-review.json';path.write_text(json.dumps(report,indent=2)+'\n');print('SAVED',path,sha(path),report['seconds'],flush=True)

if __name__=='__main__':main()
