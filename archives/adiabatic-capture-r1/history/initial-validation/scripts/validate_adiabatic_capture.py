"""Scientific checks and isolated local controls; no model-agent or Docker runs."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

BASE=Path(__file__).resolve().parents[1]
TASK=BASE/'tasks/adiabatic-capture'
RESULTS=BASE/'results'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


oracle=load('oracle',TASK/'solution/model.py')
source=load('source',BASE/'scripts/adiabatic_capture_baseline.py')
reference=load('reference',TASK/'tests/reference.py')
metadata=json.loads((TASK/'tests/metadata.json').read_text())


def nrmse(actual,truth):
    return float(np.linalg.norm(actual-truth)/np.linalg.norm(truth))


def emit(record):
    with (RESULTS/'validation-runs.jsonl').open('a') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n')
    print(json.dumps(record,allow_nan=False),flush=True)


def local_controls():
    result={}
    for name,path in [('oracle',TASK/'solution/model.py'),('shortcut',BASE/'scripts/adiabatic_capture_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='adiabatic-capture-') as folder:
            work=Path(folder)
            shutil.copytree(TASK/'environment',work/'app')
            shutil.copytree(TASK/'tests',work/'tests')
            shutil.copy2(path,work/'app/model.py')
            env=os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(work/'app'),
                       OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.monotonic()
            run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                str(work/'app/test_public.py'),str(work/'tests/test_hidden.py')],
                cwd=work,env=env,capture_output=True,text=True,timeout=60)
            result[name]={'returncode':run.returncode,'seconds':time.monotonic()-start,
                          'stdout':run.stdout,'stderr':run.stderr}
            (RESULTS/'adiabatic-capture-r1-local-controls.json').write_text(json.dumps(result,indent=2)+'\n')
            emit({'local_control':name,**result[name]})
            assert result[name]['seconds']<60
            if name=='oracle':assert run.returncode==0 and '7 passed' in run.stdout
            else:assert run.returncode==1 and '3 failed, 4 passed' in run.stdout
    return result


def run(generate=False):
    start=time.monotonic();true=reference.TRUE_PARAMETER
    inputs=reference.calibration_inputs()*metadata['calibration_repeats']
    clean=oracle.predict_at(inputs,true);sigma=metadata['measurement_sigma']
    if generate:
        assert not (TASK/'environment/data/calibration.json').exists(),'Never silently regenerate the dataset.'
        noisy=clean+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(clean))
        rows=[{'input':e,'value':float(y),'sigma':sigma} for e,y in zip(inputs,noisy)]
        data=json.dumps(rows,indent=2)+'\n'
        for side in ('environment','tests'):(TASK/side/'data/calibration.json').write_text(data)
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert len(records)==256 and all(r['sigma']==sigma for r in records)
    assert [r['input'] for r in records]==inputs
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    report={'revision':1,'status':'validation_in_progress','model_evaluations':0,
            'true_action_scale':true,'calibration_count':len(records),'calibration_settings':1,
            'sigma':sigma,'prediction_limit':.04,'reference_configuration':metadata['reference_configuration']}
    hidden=reference.hidden_inputs();truths={};per_case=[]
    cold=time.monotonic()
    for group,experiments in hidden.items():
        values=[]
        for e in experiments:
            result=reference.trajectory(true,e['delta'],e['s_final'])
            predicted=oracle.predict_at([e],true)[0]
            row={'group':group,'input':e,'reference':result,'oracle':float(predicted),
                 'relative_error':abs(result['mean_energy']/predicted-1)}
            emit({'stage':'scored_reference',**row});per_case.append(row);values.append(result['mean_energy'])
            assert row['relative_error']<.005
        truths[group]=np.array(values)
    cold=time.monotonic()-cold
    report['scored_reference_cold_seconds']=cold
    assert cold<50,'Leave time for public fitting and pytest within unchanged60s.'
    # Every scored setting receives both a joint slow/phase/action refinement and
    # an independent step refinement. No interpolation or capture rule enters ref.
    refinement=[]
    for row in per_case:
        e=row['input'];base=row['reference']['mean_energy']
        fine=reference.trajectory(true,e['delta'],e['s_final'],duration=1024.,step=.04,action_order=64,phase_order=512)
        half=reference.trajectory(true,e['delta'],e['s_final'],duration=512.,step=.02,action_order=32,phase_order=256)
        item={'input':e,'group':row['group'],'base':base,'joint_refinement':fine,'step_refinement':half,
              'relative_joint_change':abs(fine['mean_energy']/base-1),
              'relative_step_change':abs(half['mean_energy']/base-1),
              'refined_oracle_error':abs(fine['mean_energy']/row['oracle']-1)}
        emit({'stage':'per_scored_case_refinement',**item});refinement.append(item)
        assert item['relative_joint_change']<.005 and item['relative_step_change']<.0001
        assert item['refined_oracle_error']<.005
    report['scored_cases']=per_case;report['scored_refinements']=refinement
    calref=reference.predict(reference.calibration_inputs(),true)[0]
    report['calibration_reference_bias_sigma']=abs(calref-clean[0])/sigma
    assert report['calibration_reference_bias_sigma']<.001
    actual={}
    for name,module in [('oracle',oracle),('shortcut',source)]:
        model=module.Model().fit(records);p=model.predict(inputs)
        residual=(p-np.array([r['value'] for r in records]))/sigma
        actual[name]={'parameter':model.action_scale,'parameter_relative_error':abs(model.action_scale/true-1),
                      'calibration_chi2':float(residual@residual/(len(records)-1)),
                      'hidden':{g:nrmse(model.predict(v),truths[g]) for g,v in hidden.items()}}
        assert actual[name]['calibration_chi2']<1.5 and actual[name]['parameter_relative_error']<.03
    report['actual_data']=actual;emit({'stage':'actual_data','metrics':actual})
    random=np.random.default_rng(metadata['noise_seed']);extrema=[]
    max_chi=max_param=max_oracle=max_anchor=0.;min_source=100.
    for trial in range(metadata['noise_trials']):
        noisy=clean+random.normal(0,sigma,len(clean))
        sample=[{'input':e,'value':float(y),'sigma':sigma} for e,y in zip(inputs,noisy)]
        for name,module in [('oracle',oracle),('shortcut',source)]:
            model=module.Model().fit(sample);extrema.append(model.action_scale)
            pred=model.predict(inputs);chi=float(np.sum(((pred-noisy)/sigma)**2)/(len(records)-1))
            pe=abs(model.action_scale/true-1);max_chi=max(max_chi,chi);max_param=max(max_param,pe)
            assert chi<1.5 and pe<.03
            for group,experiments in hidden.items():
                error=nrmse(model.predict(experiments),truths[group])
                if name=='oracle':max_oracle=max(max_oracle,error);assert error<.04
                elif group=='connected_anchor':max_anchor=max(max_anchor,error);assert error<.04
                else:min_source=min(min_source,error);assert error>.04
        if trial%32==31:emit({'stage':'noise','completed':trial+1})
    report['noise']={'trials':256,'all_calibration_parameter_oracle_pass':True,
                     'all_shortcut_three_groups_rejected':True,'max_chi2':max_chi,'max_parameter_relative_error':max_param,
                     'max_oracle_error':max_oracle,'max_anchor_error':max_anchor,'min_shortcut_error':min_source,
                     'fit_min':min(extrema),'fit_max':max(extrema)}
    # Full parameter interval: exact noncrossing equality, global mean inverse,
    # and the same predeclared hidden groups, not a selected fitted parameter.
    fit_error=equivalence=0.;gap=100.;signal=100.;profile=[]
    for action in np.linspace(.12,.16,33):
        y=oracle.predict_at(reference.calibration_inputs(),action)[0]
        sample=[{'input':inputs[0],'value':float(y),'sigma':sigma} for _ in range(4)]
        for module in (source,oracle):fit_error=max(fit_error,abs(module.Model().fit(sample).action_scale-action))
        equivalence=max(equivalence,abs(source.predict_at(reference.calibration_inputs(),action)[0]-y))
        profile.append([float(action),float(y)])
        for g,v in hidden.items():
            exact=oracle.predict_at(v,action);approx=source.predict_at(v,action)
            if g=='connected_anchor':equivalence=max(equivalence,float(np.max(abs(exact-approx))))
            else:gap=min(gap,nrmse(approx,exact));signal=min(signal,float(np.min(abs(exact))))
    assert fit_error<1e-9 and equivalence<1e-12 and gap>.04 and signal>.5
    slopes=np.diff(np.array(profile)[:,1])/np.diff(np.array(profile)[:,0]);assert min(slopes)>0
    report['identifiability']={'proof':'For J=J0*z with fixed z uniform on[1,1.2], d<E>/dJ0=<z*omega(J0*z)> is strictly positive on all noncrossing calibration orbits.',
                              'parameter_points':33,'calibration_profile':profile,'min_sampled_derivative':float(min(slopes)),
                              'max_fit_error':fit_error,'max_calibration_anchor_equivalence':equivalence,
                              'min_hidden_group_gap':gap,'min_absolute_hidden_energy':signal}
    # Continuous declared endpoint range: quadrature accuracy away from and at
    # both edges of partial-band capture; direct checks include interior sf.
    volume_refinement=[];direct_domain=[]
    for action,delta in itertools.product([.12,.16],[.15,.25]):
        sep_coeff=oracle.geometry(delta)[2].sum()/(2*np.pi)
        endpoints=[.1,.2,.3,.45,.6,1.,1.8,2.2]
        endpoints += [(factor*action/sep_coeff)**(2/3) for factor in [.999,1.001,1.1,1.199,1.201]]
        for sf in endpoints:
            e={'delta':delta,'s_final':float(sf)}
            coarse=oracle.mean_energy(action,delta,float(sf),32)
            fine=oracle.mean_energy(action,delta,float(sf),64)
            err=abs(coarse-fine)/max(1.,abs(fine));volume_refinement.append(err)
            assert err<2e-7
        for sf in [.1,.3,.6,2.2]:
            exact=oracle.predict_at([{'delta':delta,'s_final':sf}],action)[0]
            ref=reference.trajectory(action,delta,sf)
            err=abs(ref['mean_energy']-exact)/max(1.,abs(exact))
            row={'parameter':action,'delta':delta,'s_final':sf,'oracle':float(exact),'reference':ref,'scaled_error':err}
            direct_domain.append(row);emit({'stage':'domain_reference',**row})
            assert err<.003
    report['full_domain']={'static_count':len(volume_refinement),'max_order32_to64_scaled_change':max(volume_refinement),
                           'direct_cases':direct_domain,'max_direct_scaled_error':max(r['scaled_error'] for r in direct_domain)}
    symmetric=[]
    for sf in [.1,.5,1.8,2.2]:
        exact=oracle.mean_energy(true,0.,sf);approx=source.mean_energy(true,0.,sf)
        symmetric.append(abs(exact-approx));assert abs(exact-approx)<1e-10
    report['limits']={'symmetric_max_difference':max(symmetric),
                       'partial_capture_reference':'prototype/partial-report.json; exact half-band threshold, direct duration/grid refinement preserved',
                       'energy_work_max_abs':max(r['reference']['energy_work_max_abs'] for r in per_case+direct_domain),
                       'initial_energy_max_abs':max(r['reference']['initial_energy_max_abs'] for r in per_case+direct_domain)}
    report['local_controls']=local_controls()
    report['status']='scientific_and_local_controls_complete';report['seconds']=time.monotonic()-start
    paths=list(TASK.rglob('*'))+list((BASE/'scripts').glob('*.py'))
    report['source_hashes']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file() and '__pycache__' not in p.parts}
    (RESULTS/'adiabatic-capture-r1-validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    emit({'stage':'complete','seconds':report['seconds'],'noise':report['noise'],'identifiability':report['identifiability']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    try:run(args.generate)
    except Exception as error:
        emit({'stage':'validation_exception','type':type(error).__name__,'message':str(error)})
        raise
