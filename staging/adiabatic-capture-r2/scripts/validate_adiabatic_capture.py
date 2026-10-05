"""Changing-path capture science and isolated local controls; no model launches."""
import argparse
import ast
from itertools import product
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import numpy as np

BASE=Path(__file__).resolve().parents[1]
TASK=BASE/'tasks/adiabatic-capture'
RESULTS=BASE/'results'
REPORT={}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


oracle=load('path_oracle',TASK/'solution/model.py')
source=load('path_source',BASE/'scripts/adiabatic_capture_baseline.py')
reference=load('path_reference',TASK/'tests/reference.py')
metadata=json.loads((TASK/'tests/metadata.json').read_text())


def nrmse(x,y):
    return float(np.sqrt(np.mean((x-y)**2)/np.mean(y*y)))


def direct(action_scale,e,label,**configuration):
    result=reference.trajectory(float(action_scale),float(e['delta_final']),float(e['path_slope']),
                                float(e['s_final']),**configuration)
    correct=float(oracle.predict_at([e],action_scale)[0])
    row={'label':label,'action_scale':float(action_scale),'input':e,'configuration':configuration,
         'trajectory':result,'slow_limit':correct,'relative_error':abs(result['mean_energy']/correct-1)}
    with (RESULTS/'validation-runs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps({'stage':label,'input':e,'relative_error':row['relative_error'],'seconds':result['seconds']}),flush=True)
    return row


def local_controls():
    outcomes={}
    for name,path in [('oracle',TASK/'solution/model.py'),('shortcut',BASE/'scripts/adiabatic_capture_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='adiabatic-path-control-') as folder:
            root=Path(folder);shutil.copytree(TASK/'environment',root/'app');shutil.copytree(TASK/'tests',root/'tests')
            shutil.copy2(path,root/'app/model.py')
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'app'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            begin=time.monotonic()
            run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(root/'app/test_public.py'),str(root/'tests/test_hidden.py')],cwd=root,env=env,capture_output=True,text=True,timeout=60)
            outcomes[name]={'returncode':run.returncode,'seconds':time.monotonic()-begin,'stdout':run.stdout,'stderr':run.stderr}
    (RESULTS/'adiabatic-capture-r2-local-controls.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    assert outcomes['oracle']['returncode']==0 and '7 passed' in outcomes['oracle']['stdout']
    assert outcomes['shortcut']['returncode']==1 and '3 failed, 4 passed' in outcomes['shortcut']['stdout']
    assert all(row['seconds']<60 for row in outcomes.values())
    return {k:{a:b for a,b in v.items() if a not in ('stdout','stderr')} for k,v in outcomes.items()}


def run(generate=False):
    begin=time.monotonic();truth=reference.TRUE_PARAMETER;sigma=metadata['measurement_sigma']
    settings=reference.calibration_inputs();inputs=settings*metadata['calibration_repeats']
    clean=oracle.predict_at(inputs,truth)
    if generate:
        values=clean+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        rows=[{'input':e,'value':float(y),'sigma':sigma} for e,y in zip(inputs,values)]
        text=json.dumps(rows,indent=2)+'\n'
        for side in ('environment','tests'):(TASK/side/'data/calibration.json').write_text(text)
    rows=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert len(rows)==256 and [r['input'] for r in rows]==inputs and all(r['sigma']==.001 for r in rows)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    REPORT.update(revision=2,status='science_in_progress',model_runs=0,
                  calibration={'settings':1,'repeats':256,'records':256,'sigma':sigma,
                   'generation':'Exact classical slow-limit area calculation; finite-duration reference bias is separate from instrument noise.',
                   'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed']})
    functions=lambda path:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    assert functions(TASK/'environment/model.py')==functions(BASE/'scripts/adiabatic_capture_baseline.py')
    groups=reference.hidden_inputs();scored=[]
    for name,experiments in groups.items():
        for e in experiments:
            original=direct(truth,e,'scored_default')
            slow=direct(truth,e,'scored_duration_refinement',duration=1024.)
            dense=direct(truth,e,'scored_ensemble_refinement',action_order=48,phase_order=384)
            scored.append({'group':name,'input':e,'default':original,'slower':slow,'denser':dense,
                           'duration_change_relative':abs(slow['trajectory']['mean_energy']-original['trajectory']['mean_energy'])/abs(original['slow_limit']),
                           'ensemble_change_relative':abs(dense['trajectory']['mean_energy']-original['trajectory']['mean_energy'])/abs(original['slow_limit'])})
    REPORT['scored_reference']={'rows':scored,
        'maximum_default_relative_error':max(r['default']['relative_error'] for r in scored),
        'maximum_slower_relative_error':max(r['slower']['relative_error'] for r in scored),
        'maximum_denser_relative_error':max(r['denser']['relative_error'] for r in scored),
        'maximum_duration_change_relative':max(r['duration_change_relative'] for r in scored),
        'maximum_ensemble_change_relative':max(r['ensemble_change_relative'] for r in scored)}
    assert max(REPORT['scored_reference'][key] for key in REPORT['scored_reference'] if key!='rows')<.006
    calibration_checks=[direct(truth,settings[0],'calibration_finite_ramp',duration=t) for t in [512.,1024.]]
    REPORT['calibration']['finite_duration_checks']=calibration_checks
    REPORT['calibration']['scope']='These finite-ramp errors are deterministic reference error; data generation uses the exact slow limit and does not count them as measurement noise.'
    truths={k:reference.predict(v,truth) for k,v in groups.items()}
    actual={}
    for name,module in [('oracle',oracle),('shortcut',source)]:
        fitted=module.Model().fit(rows);residual=(fitted.predict(inputs)-np.array([r['value'] for r in rows]))/sigma
        actual[name]={'action_scale':fitted.action_scale,'parameter_relative_error':abs(fitted.action_scale/truth-1),
                      'calibration_chi2':float(residual@residual)/(len(rows)-1),
                      'hidden':{k:nrmse(fitted.predict(v),truths[k]) for k,v in groups.items()}}
    REPORT['actual_data']=actual
    noise=np.random.default_rng(metadata['noise_seed']);fits=[];max_oracle=max_parameter=max_chi=max_anchor=0.;min_source=float('inf');max_fit_difference=0.
    for trial in range(metadata['noise_trials']):
        values=clean+noise.normal(0,sigma,len(inputs))
        noisy=[{'input':e,'value':float(y),'sigma':sigma} for e,y in zip(inputs,values)]
        models=[(name,module.Model().fit(noisy)) for name,module in [('oracle',oracle),('shortcut',source)]]
        max_fit_difference=max(max_fit_difference,abs(models[0][1].action_scale-models[1][1].action_scale))
        for name,model in models:
            fits.append(model.action_scale)
            chi=float(np.sum(((model.predict(inputs)-values)/sigma)**2)/(len(inputs)-1))
            parameter=abs(model.action_scale/truth-1);max_chi=max(max_chi,chi);max_parameter=max(max_parameter,parameter)
            assert chi<1.5 and parameter<.03
            for k,v in groups.items():
                error=nrmse(model.predict(v),truths[k])
                if name=='oracle':max_oracle=max(max_oracle,error);assert error<.04
                elif k=='constant_path_anchor':max_anchor=max(max_anchor,error);assert error<.04
                else:min_source=min(min_source,error);assert error>.04
        if trial%64==63:print(json.dumps({'stage':'noise','completed':trial+1}),flush=True)
    REPORT['noise']={'trials':metadata['noise_trials'],'all_expected_outcomes':True,'fit_min':min(fits),'fit_max':max(fits),
                    'maximum_fit_difference':max_fit_difference,'maximum_parameter_relative_error':max_parameter,
                    'maximum_chi2':max_chi,'maximum_oracle_error':max_oracle,'maximum_source_anchor_error':max_anchor,
                    'minimum_source_diagnostic_error':min_source}
    recovered=[];gap={k:float('inf') for k in groups if k!='constant_path_anchor'};max_shared=0.
    for parameter in np.linspace(.12,.16,41):
        value=float(oracle.predict_at(settings,parameter)[0]);records=[{'input':settings[0],'value':value,'sigma':sigma}]
        estimates=[module.Model().fit(records).action_scale for module in (oracle,source)]
        recovered.append({'truth':float(parameter),'estimates':estimates})
        max_shared=max(max_shared,abs(value-source.predict_at(settings,parameter)[0]))
        for k,v in groups.items():
            if k!='constant_path_anchor':gap[k]=min(gap[k],nrmse(source.predict_at(v,parameter),oracle.predict_at(v,parameter)))
    REPORT['identifiability']={'proof':'At a=0, d(mean E)/dJ0=mean(sum_i alpha_i^2*xi*omega_i)>0; branch frequencies are positive. This proves global injectivity for the captured calibration. No claim is made about all noisy objective landscapes.',
     'parameter_samples':41,'recoveries':recovered,'maximum_fit_error':max(abs(row['truth']-v) for row in recovered for v in row['estimates']),
     'maximum_shared_calibration_error':float(max_shared),'diagnostic_group_gap_minima':gap}
    assert REPORT['identifiability']['maximum_fit_error']<1e-9 and max_shared<1e-12 and min(gap.values())>.07
    domain=list(product([.12,.14,.16],[.15,.2,.25],[0.,.04,.08,.12],[1.8,2.,2.2]))
    random=np.random.default_rng(172027)
    domain += [(random.uniform(.12,.16),random.uniform(.15,.25),random.uniform(0,.12),random.uniform(1.8,2.2)) for _ in range(24)]
    domain_rows=[];max_quad=0.;min_growth=float('inf');max_derivative=0.
    for j,delta,a,sf in domain:
        e={'delta_final':float(delta),'path_slope':float(a),'s_final':float(sf)}
        physical=oracle.mean_energy(j,delta,a,sf,24);fine=oracle.mean_energy(j,delta,a,sf,48)
        wrong=source.mean_energy(j,delta,a,sf,24);wrong_fine=source.mean_energy(j,delta,a,sf,48)
        max_quad=max(max_quad,abs(physical-fine),abs(wrong-wrong_fine))
        assert np.isfinite([physical,fine,wrong,wrong_fine]).all() and physical<0 and wrong<0
        if a==0:assert abs(physical-wrong)<1e-12
        for s in np.geomspace(.05,sf,33):
            d=delta+a*np.log(s/sf);c=oracle.geometry(float(d))[0][1];areas=oracle.geometry(float(d))[2]
            ang=np.arcsin(2*c/np.sqrt(2*(1-c*c)));deriv=2*np.sqrt(2)*np.array([ang+np.pi/2,ang-np.pi/2])
            rates=1.5*areas+a*deriv;min_growth=min(min_growth,float(np.sqrt(s)*rates.min()))
            numerical=(oracle.geometry(float(d+1e-5))[2]-oracle.geometry(float(d-1e-5))[2])/(2e-5)
            max_derivative=max(max_derivative,float(abs(numerical-deriv).max()))
            assert rates.min()>0
        domain_rows.append({'action_scale':float(j),'input':e,'physical':physical,'source':wrong,'relative_gap':abs(wrong/physical-1)})
    REPORT['domain']={'cases':len(domain),'rows':domain_rows,'maximum_quadrature_change':max_quad,
                     'minimum_sampled_lobe_growth':min_growth,'maximum_area_derivative_error':max_derivative,
                     'growth_proof':'For saddle c in [-.35,.27], A_R prime<=-3.28 and A_R second<9.1. Thus partial_delta(1.5*A_R+a*A_R prime)<-3.82 and partial_a<0. Minimum bracket at delta=.25,a=.12 is .139270910426599>0. Left growth is positive directly.',
                     'qualification':'Source agrees at zero slope and approaches the oracle near zero; no uniform discrepancy is claimed on the entire public domain.'}
    assert max_quad<1e-9 and max_derivative<1e-7
    edges=[]
    for j,delta,a,sf in [(.12,.15,0.,1.8),(.16,.25,0.,2.2),(.12,.15,.12,2.2),(.16,.25,.12,1.8),(.12,.25,.08,2.),(.16,.15,.04,1.8)]:
        row=direct(j,{'delta_final':delta,'path_slope':a,'s_final':sf},'parameter_domain_direct',duration=1024.)
        edges.append(row);assert row['relative_error']<.006
    diagnostic={'delta_final':.2,'path_slope':.12,'s_final':2.2}
    step=direct(truth,diagnostic,'scored_half_step',step=.02)
    phase=direct(truth,diagnostic,'scored_phase_offset',phase_offset=.17)
    default=reference.trajectory(truth,.2,.12,2.2)
    REPORT['direct_domain_and_limits']={'edge_rows':edges,'half_step':step,'phase_offset':phase,
        'half_step_mean_change':abs(step['trajectory']['mean_energy']-default['mean_energy']),
        'phase_offset_mean_change':abs(phase['trajectory']['mean_energy']-default['mean_energy']),
        'maximum_edge_relative_error':max(row['relative_error'] for row in edges),
        'zero_slope_equality_and_outgoing_action_preservation':True}
    assert REPORT['direct_domain_and_limits']['half_step_mean_change']<1e-5
    assert REPORT['direct_domain_and_limits']['phase_offset_mean_change']<.005
    REPORT['local_controls']=local_controls()
    REPORT['source_sha256']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',BASE/'scripts/adiabatic_capture_baseline.py',Path(__file__)]}
    REPORT['status']='science_and_local_controls_complete';REPORT['seconds']=time.monotonic()-begin
    (RESULTS/'adiabatic-capture-r2-validation.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:REPORT[k] for k in ['status','actual_data','noise','local_controls','seconds']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true')
    try:run(parser.parse_args().generate)
    except Exception:
        REPORT['status']='author_check_failed';REPORT['traceback']=traceback.format_exc()
        (RESULTS/f'author-check-failure-{time.time_ns()}.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
        raise
