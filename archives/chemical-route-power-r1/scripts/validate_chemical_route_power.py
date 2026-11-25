"""Scientific validation and local controls; no model-agent evaluations."""
import argparse
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
import numpy as np

BASE = Path(__file__).resolve().parents[1]
TASK = BASE/'tasks/chemical-route-power'
RESULTS = BASE/'results'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load('oracle', TASK/'solution/model.py')
shortcut = load('shortcut', BASE/'scripts/chemical_route_power_baseline.py')
reference = load('physical_reference', TASK/'tests/reference.py')
metadata = json.loads((TASK/'tests/metadata.json').read_text())


def nrmse(value, truth):
    return float(np.sqrt(np.mean((value-truth)**2)/np.mean(truth**2)))


def state_check(e):
    energies=np.array(e['energies']);v=np.array(e['attempts']);a=np.array(e['affinities'])
    d=np.roll(energies,-1)-energies
    forward=v*np.exp((a[None,:]-d[:,None])/2)
    backward=v*np.exp(-(a[None,:]-d[:,None])/2)
    generator=np.zeros((3,3))
    for i in range(3):
        j=(i+1)%3
        generator[j,i]+=sum(forward[i]);generator[i,j]+=sum(backward[i])
        generator[i,i]-=sum(forward[i]);generator[j,j]-=sum(backward[i])
    system=generator.copy();system[-1]=1
    p=np.linalg.solve(system,[0.,0.,1.])
    f=p[:,None]*forward;b=np.roll(p,-1)[:,None]*backward
    j=f-b
    fine=float(np.sum(j*np.log(f/b)))
    energy=float(np.sum(j*d[:,None]))
    aggregated_f=f.sum(axis=1);aggregated_b=b.sum(axis=1)
    coarse=float(np.sum((aggregated_f-aggregated_b)*np.log(aggregated_f/aggregated_b)))
    fp=f/aggregated_f[:,None];bp=b/aggregated_b[:,None]
    kl=float(np.sum(f*np.log(fp/bp)+b*np.log(bp/fp)))
    return p,generator,energy,fine,coarse,kl


def domain_cases(low_attempt=.5, high_attempt=1.5, diagnostic=False):
    affinities=[[1.6,2.4],[.2,.6]] if diagnostic else [[.2,2.4],[.2,2.4]]
    for e1,e2,a0,a1,*v in product(*([[-.3,.3]]*2+affinities+[[low_attempt,high_attempt]]*6)):
        yield {'energies':[0.,e1,e2], 'affinities':[a0,a1],
               'attempts':np.array(v).reshape(3,2).tolist()}


def local_controls():
    report={}
    for name,path in [('oracle',TASK/'solution/model.py'),
                      ('shortcut',BASE/'scripts/chemical_route_power_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='chemical-route-controls-') as folder:
            location=Path(folder)
            shutil.copytree(TASK/'environment',location/'app')
            shutil.copytree(TASK/'tests',location/'tests')
            shutil.copy2(path,location/'app/model.py')
            env=os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(location/'app'),
                       OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.monotonic()
            run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                                str(location/'app/test_public.py'),str(location/'tests/test_hidden.py')],
                               cwd=location,env=env,capture_output=True,text=True,timeout=60)
            report[name]={'returncode':run.returncode,'seconds':time.monotonic()-start,
                          'stdout':run.stdout,'stderr':run.stderr}
            if name=='oracle': assert run.returncode==0 and '7 passed' in run.stdout
            else: assert run.returncode==1 and '3 failed, 4 passed' in run.stdout
    (RESULTS/'chemical-route-power-r1-local-controls.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:{kk:vv for kk,vv in v.items() if kk not in ('stdout','stderr')} for k,v in report.items()}


def run(generate=False):
    start=time.monotonic()
    inputs=reference.calibration_inputs()*metadata['calibration_repeats']
    true=reference.TRUE_PARAMETER
    clean=reference.predict(inputs,true)
    sigma=metadata['measurement_sigma']
    if generate:
        measured=clean+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        rows=[{'input':e,'value':float(y),'sigma':sigma} for e,y in zip(inputs,measured)]
        data=json.dumps(rows,indent=2)+'\n'
        for side in ('environment','tests'):(TASK/side/'data/calibration.json').write_text(data)
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    assert len(records)==288
    assert all(r['sigma']==sigma for r in records)
    assert [r['input'] for r in records]==inputs
    report={'revision':1,'status':'scientific_and_local_controls_complete','model_evaluations':0,
            'parameter':true,'calibration_seed':metadata['calibration_seed'],
            'noise_seed':metadata['noise_seed'],'calibration_count':len(records),
            'calibration_distinct_settings':len(reference.calibration_inputs()),
            'sigma':sigma,'prediction_limit':metadata['prediction_limit']}
    hidden=reference.hidden_inputs()
    truths={k:reference.predict(v,true) for k,v in hidden.items()}
    ref_error=0.;ref_refinement=0.
    for k,v in hidden.items():
        ref_error=max(ref_error,float(np.max(abs(oracle.predict_at(v,true)-truths[k]))))
        ref_refinement=max(ref_refinement,float(np.max(abs(reference.predict(v,true,5e-6)-truths[k]))))
    results={}
    for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
        model=mod.Model().fit(records)
        y=model.predict(inputs)
        residual=(y-np.array([r['value'] for r in records]))/sigma
        results[name]={'parameter':model.rate_scale,'parameter_relative_error':abs(model.rate_scale/true-1),
                       'calibration_chi2':float(residual@residual/(len(records)-1)),
                       'hidden':{k:nrmse(model.predict(v),truths[k]) for k,v in hidden.items()}}
        assert results[name]['parameter_relative_error']<.03 and results[name]['calibration_chi2']<1.5
    report['actual_data']=results
    report['reference_hidden']={'max_absolute_error':ref_error,'max_half_step_change':ref_refinement,
                                'min_signal':float(min(np.min(abs(x)) for x in truths.values())),
                                'calibration_bias_sigma':float(np.max(abs(oracle.predict_at(inputs,true)-clean))/sigma)}
    assert ref_error<1e-7 and ref_refinement<1e-7
    # Both completed fit methods actually execute for all noise realizations.
    noise=np.random.default_rng(metadata['noise_seed'])
    extrema=[];maximum_chi=maximum_param=maximum_oracle=maximum_anchor=0.;minimum_shortcut=1.
    for _ in range(metadata['noise_trials']):
        noisy=clean+noise.normal(0,sigma,len(clean))
        sample=[{'input':e,'value':float(y),'sigma':sigma} for e,y in zip(inputs,noisy)]
        for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
            model=mod.Model().fit(sample);parameter=model.rate_scale
            pred=model.predict(inputs);chi=float(np.sum(((pred-noisy)/sigma)**2)/(len(inputs)-1))
            param=abs(parameter/true-1)
            maximum_chi=max(maximum_chi,chi);maximum_param=max(maximum_param,param)
            assert chi<1.5 and param<.03
            for group,experiments in hidden.items():
                error=nrmse(model.predict(experiments),truths[group])
                if name=='oracle':
                    maximum_oracle=max(maximum_oracle,error);assert error<.04
                elif group=='equal_drive_anchor':
                    maximum_anchor=max(maximum_anchor,error);assert error<.04
                else:
                    minimum_shortcut=min(minimum_shortcut,error);assert error>.04
            extrema.append(parameter)
    report['noise']={'trials':metadata['noise_trials'],'all_expected_outcomes':True,
                     'parameter_min':min(extrema),'parameter_max':max(extrema),
                     'max_calibration_chi2':maximum_chi,'max_parameter_relative_error':maximum_param,
                     'max_oracle_hidden_error':maximum_oracle,'min_shortcut_diagnostic_error':minimum_shortcut,
                     'max_shortcut_anchor_error':maximum_anchor}
    # Exact calibration identity/global identification across the full unknown interval.
    unit=oracle.predict_at(inputs,1.)
    cal_equivalence=recovery_error=0.;domain_gap=1.
    for scale in np.linspace(.6,1.4,41):
        y=oracle.predict_at(inputs,scale)
        cal_equivalence=max(cal_equivalence,float(np.max(abs(y-shortcut.predict_at(inputs,scale)))))
        rows=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,y)]
        for mod in (oracle,shortcut): recovery_error=max(recovery_error,abs(mod.Model().fit(rows).rate_scale-scale))
        for group,experiments in hidden.items():
            if group!='equal_drive_anchor':
                domain_gap=min(domain_gap,nrmse(shortcut.predict_at(experiments,scale),oracle.predict_at(experiments,scale)))
    report['identifiability']={'proof':'Every power equals rate_scale times its positive unit-scale value, so the weighted calibration objective is strictly convex.',
                               'min_derivative':float(unit.min()),'weighted_curvature':float(2*np.sum((unit/sigma)**2)),
                               'scale_samples':41,'max_noiseless_recovery_error':recovery_error,
                               'max_calibration_equivalence_error':cal_equivalence,
                               'min_hidden_gap_across_scale':domain_gap}
    assert cal_equivalence<1e-12 and recovery_error<1e-12 and domain_gap>.04
    # Full PUBLIC energy/barrier/affinity corner box, not just the diagnostic subset.
    corners=list(domain_cases())
    min_p=1.;max_stationary=max_energy=max_entropy=max_logsum=0.;min_power=min_aggregate=1e9
    for e in corners:
        p,g,energy,fine,coarse,kl=state_check(e)
        physical=oracle.predict_at([e],1.)[0]
        min_p=min(min_p,float(p.min()));max_stationary=max(max_stationary,float(np.max(abs(g@p))))
        max_energy=max(max_energy,abs(energy));max_entropy=max(max_entropy,abs(physical-fine))
        max_logsum=max(max_logsum,abs((fine-coarse)-kl))
        min_power=min(min_power,physical);min_aggregate=min(min_aggregate,coarse)
        assert kl>-1e-12 and physical>=coarse-1e-12
    random=np.random.default_rng(148043)
    reference_cases=corners[::16]+[{'energies':[0.,*random.uniform(-.3,.3,2)],
                                  'attempts':random.uniform(.5,1.5,(3,2)).tolist(),
                                  'affinities':random.uniform(.2,2.4,2).tolist()} for _ in range(64)]
    ref_domain=ref_domain_step=0.
    for i,e in enumerate(reference_cases):
        scale=.6 if i%2 else 1.4
        value=oracle.predict_at([e],scale)[0];r=reference.predict([e],scale)[0]
        ref_domain=max(ref_domain,abs(value-r)/max(1.,abs(value)))
        ref_domain_step=max(ref_domain_step,abs(reference.predict([e],scale,5e-6)[0]-r)/max(1.,abs(value)))
    report['full_public_domain']={'corner_count':len(corners),'attempt_range':[.5,1.5],
                                  'affinity_ranges':[[.2,2.4],[.2,2.4]],'min_probability':min_p,
                                  'min_power_at_scale1':min_power,'min_aggregate_power_at_scale1':min_aggregate,
                                  'max_probability_balance':max_stationary,'max_energy_balance':max_energy,
                                  'max_chemical_power_entropy_identity':max_entropy,'max_logsum_identity':max_logsum,
                                  'independent_reference_count':len(reference_cases),
                                  'max_scaled_reference_error':ref_domain,'max_half_step_change':ref_domain_step}
    assert min_p>0 and min_power>0 and min_aggregate>0 and ref_domain<1e-7
    assert max(max_stationary,max_energy,max_entropy,max_logsum)<1e-12
    # Symmetries, equilibrium and rate scaling at off-grid controls.
    limits={'max_energy_shift_error':0.,'max_channel_exchange_error':0.,'max_rate_scaling_error':0.,'max_equilibrium_power':0.}
    for e in reference_cases[-32:]:
        shifted={**e,'energies':(np.array(e['energies'])+5.).tolist()}
        swapped={**e,'attempts':np.array(e['attempts'])[:,::-1].tolist(),'affinities':e['affinities'][::-1]}
        equilibrium={**e,'affinities':[0.,0.]}
        for mod in (oracle,shortcut):
            value=mod.predict_at([e],1.)[0]
            limits['max_energy_shift_error']=max(limits['max_energy_shift_error'],abs(value-mod.predict_at([shifted],1.)[0]))
            limits['max_channel_exchange_error']=max(limits['max_channel_exchange_error'],abs(value-mod.predict_at([swapped],1.)[0]))
            limits['max_rate_scaling_error']=max(limits['max_rate_scaling_error'],abs(1.3*value-mod.predict_at([e],1.3)[0]))
            limits['max_equilibrium_power']=max(limits['max_equilibrium_power'],abs(mod.predict_at([equilibrium],1.)[0]))
    assert max(limits.values())<1e-12
    report['limits']=limits
    report['local_controls']=local_controls()
    report['seconds']=time.monotonic()-start
    report['source_hashes']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',
        BASE/'scripts/chemical_route_power_baseline.py',Path(__file__)]}
    (RESULTS/'chemical-route-power-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    run(parser.parse_args().generate)
