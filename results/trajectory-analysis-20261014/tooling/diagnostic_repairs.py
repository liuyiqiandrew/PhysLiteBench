from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, types, sys
import numpy as np

R=Path('/Users/yp9846/Desktop/PhysLiteBench')
A=R/'results/trajectory-analysis-20261014'
D=json.loads((A/'analysis.json').read_text())

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def module(code,name):
    m=types.ModuleType(name);exec(compile(code,'<diagnostic-memory-copy>','exec'),m.__dict__);return m
def replace_once(code,old,new):
    assert code.count(old)==1,(old,code.count(old))
    return code.replace(old,new)
def err(actual,truth):
    actual,truth=np.asarray(actual),np.asarray(truth)
    assert actual.shape==truth.shape and np.isfinite(actual).all()
    return float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))

rows=[]
for tid in ['entropy-anomaly__6yBuZGL','magnetic-tracer__GAWaaTf','single-file-memory__PC6LumP','reaction-diffusion__2xoVywU']:
    r=next(r for r in D['trials'] if r['trial']==tid)
    p=R/r['path'];f=p/'artifacts/app/model.py';frozen=p.parent/'frozen-task'
    metrics=json.loads((p/'verifier/metrics.json').read_text());code=f.read_text();orig=module(code,'original')
    old=new=None
    if r['task']=='entropy-anomaly':
        old='density_correction={(0,0):np.zeros((count,points))}'
        new='contact_flux=d@first[1,0].T\n    q_aug=q.copy();q_aug[-1]=1.\n    source=contact_flux.T.copy();source[-1]=0.\n    population_correction=np.linalg.solve(q_aug,source)\n    density_correction={(0,0):population_correction}'
        repaired=module(replace_once(code,old,new),'repaired')
        param=metrics['parameter']; key='friction'
        groups=json.loads((frozen/'tests/data/hidden_truth.json').read_text())['groups']
        description='Restore the first-order conditional contact population from the contact continuity equation.'
    elif r['task']=='magnetic-tracer':
        # All other functions, the fitted parameter and discretization stay fixed.
        fixed=replace_once(code,'drift=[];diff=[];g=[];h=[]','drift=[];diff=[];g=[];h=[];cross=[]')
        fixed=replace_once(fixed,'h.append(weight[j]**2*D[1,1])','h.append(weight[j]**2*D[1,1]);cross.append(weight[j]*D[0,1])')
        fixed=replace_once(fixed,'A1=np.diag(g);A2','A1=np.diag(g)-2*dx@np.diag(cross);A2')
        repaired=module(fixed,'repaired');param=metrics['parameters']['friction'];key='friction'
        groups=json.loads((frozen/'tests/data/hidden_truth.json').read_text())['groups']
        description='Restore the mixed longitudinal/detector diffusion term in the tilted generator.'
    elif r['task']=='single-file-memory':
        old='time_factor = np.sqrt(time_a) + np.sqrt(time_b) - np.sqrt(abs(time_a - time_b))'
        new='time_factor = np.sqrt(2.0) * (np.sqrt(time_a + time_b) - np.sqrt(abs(time_a - time_b)))'
        repaired=module(replace_once(code,old,new),'repaired');param=metrics['parameter'];key='diffusivity'
        ref=load(frozen/'tests/reference.py','single_file_reference')
        groups={name:dict(inputs=es,truth=ref.predict(es,ref.TRUE_PARAMETER)) for name,es in ref.hidden_inputs().items()}
        description='Replace the stationary-increment covariance with the deterministic-preparation two-time covariance.'
    else:
        old='@ np.diag(totals)';new='@ susceptibility'
        assert code.count(old)==2  # Public correlate method and compatibility helper.
        repaired=module(code.replace(old,new),'repaired');param=metrics['diffusivity_m2_s'];key='diffusivity'
        ref=load(frozen/'tests/reference.py','reaction_reference')
        groups={name:dict(inputs=es,truth=np.concatenate([ref.correlations(**e).ravel() for e in es])) for name,es in ref.fluctuation_cases().items()}
        description='Use the existing charge-conditioned equilibrium susceptibility as the initial mode covariance instead of independent Poisson total counts.'
    before={};after={}
    for name,group in groups.items():
        if r['task']=='reaction-diffusion':
            def predict(m):
                obj=m.TransportModel();obj.diffusivity=param
                return np.concatenate([obj.correlate(e['t'],e['totals'],e['harmonic']).ravel() for e in group['inputs']])
        else:
            def predict(m):
                obj=m.Model();setattr(obj,key,param);return obj.predict(group['inputs'])
        before[name]=err(predict(orig),group['truth']);after[name]=err(predict(repaired),group['truth'])
    prior=metrics['fluctuation_nrmse'] if r['task']=='reaction-diffusion' else metrics['hidden']
    delta=max(abs(before[k]-prior[k]) for k in before)
    assert delta<2e-6,(tid,delta)
    limit=.05 if r['task']=='reaction-diffusion' else .04
    assert max(after.values())<limit,(tid,after)
    assert sha(f)==r['source_sha256']
    row=dict(trial=tid,task=r['task'],model=r['model'],effort=r['reasoning_effort'],intervention=description,parameter_held_fixed=param,original_source_sha256=sha(f),before=before,after=after,prediction_limit=limit,max_original_metric_reproduction_difference=delta,all_tested_prediction_groups_under_limit=True,original_reward_unchanged=r['reward'],diagnostic_only=True)
    rows.append(row);print(json.dumps(row),flush=True)
(A/'diagnostic-repairs.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),method='Post-hoc in-memory diagnostic interventions. No new model attempts, refitting, source edits, reward changes or benchmark regrading. Each original group error reproduced within 2e-6 before intervention. Reaction diffusion intervention evaluates the three fluctuation groups; all other cases evaluate every hidden prediction group. These analyst-provided repairs diagnose the submission and do not count as model successes.',numpy=np.__version__,cases=rows),indent=2)+'\n')
