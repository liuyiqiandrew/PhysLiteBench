"""Scientific validation of the staged radiation task; no model evaluations."""
import argparse
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

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/radiative-angular-closure'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def moment_flux(state):
    energy=state[0];flux=state[1:];f=np.linalg.norm(flux)/energy
    if f==0:pressure=energy*np.eye(3)/3
    else:
        chi=(3+4*f*f)/(5+2*np.sqrt(4-3*f*f))
        n=flux/np.linalg.norm(flux)
        pressure=energy*((1-chi)*np.eye(3)/2+(3*chi-1)*np.outer(n,n)/2)
    return np.r_[flux[0],pressure[0]]


def real_grid_response(e,absorption,n):
    """Second-order conservative real-space finite-volume transport."""
    dx=2*np.pi/n;x=np.arange(n)*dx
    velocity=np.array(e['directions'])[:,0]
    field=np.array(e['weights'])[:,None]*np.array(e['modulations'])[:,None]*np.cos(e['wavenumber']*x)
    steps=max(1,int(np.ceil(e['time']*max(np.max(abs(velocity)),.1)/(.6*dx))))
    dt=e['time']/steps;c=velocity[:,None]*dt/dx
    for _ in range(steps):
        left=np.roll(field,1,axis=1);right=np.roll(field,-1,axis=1)
        field=field-c*(right-left)/2+c*c*(right-2*field+left)/2
    signal=np.sum(field,axis=0)*np.exp(-absorption*e['time'])
    position=(e['position']%(2*np.pi))/dx;i=int(position);fraction=position-i
    return (1-fraction)*signal[i]+fraction*signal[(i+1)%n]


def science(good,bad,ref):
    rng=np.random.default_rng(109381);errors=[];fd=[];imag=[];speed=[]
    for _ in range(96):
        rays=rng.normal(size=(int(rng.integers(2,7)),3));rays/=np.linalg.norm(rays,axis=1)[:,None]
        weights=rng.uniform(.1,1,len(rays));weights/=sum(weights)
        e=dict(directions=rays.tolist(),weights=weights.tolist(),modulations=rng.uniform(-1,1,len(rays)).tolist(),wavenumber=int(rng.integers(0,4)),time=float(rng.uniform(0,3.2)),position=float(rng.uniform(0,2*np.pi)))
        absorption=float(rng.uniform(.12,.45));a=good.Model();a.absorption=absorption
        errors.append(abs(a.predict([e])[0]-ref.predict([e],absorption)[0]))
        state=np.r_[sum(weights),weights@rays];matrix=bad.flux_jacobian(state[0],state[1:]);h=1e-6
        numerical=np.column_stack([(moment_flux(state+h*np.eye(4)[i])-moment_flux(state-h*np.eye(4)[i]))/(2*h) for i in range(4)])
        fd.append(np.max(abs(numerical-matrix)))
        eig=np.linalg.eigvals(matrix);imag.append(max(abs(eig.imag)));speed.append(max(abs(eig.real)))
    zero=bad.flux_jacobian(1,np.zeros(3));expected=np.zeros((4,4));expected[0,1]=1;expected[1,0]=1/3
    assert np.array_equal(zero,expected)
    tiny=max(np.max(abs(bad.flux_jacobian(1,np.array([f,0,0]))-zero)) for f in [1e-14,1e-10])
    assert max(errors)<1e-12 and max(fd)<2e-8 and max(imag)<1e-10 and max(speed)<=1+1e-10 and tiny<1e-8
    hidden=ref.hidden_inputs();all_inputs=sum(hidden.values(),[])
    for e in all_inputs+ref.calibration_inputs():
        assert 2<=len(e['directions'])<=6 and np.max(abs(np.linalg.norm(e['directions'],axis=1)-1))<1e-12
        assert min(e['weights'])>0 and abs(sum(e['weights'])-1)<1e-12
        assert max(abs(np.array(e['modulations'])))<=1 and 0<=e['time']<=3.2
    a=good.Model();b=bad.Model();a.absorption=b.absorption=0
    limit_e=dict(directions=[[1,0,0],[-1,0,0]],weights=[.5,.5],modulations=[.7,.7],wavenumber=2,time=.9,position=.23)
    true_limit=.7*np.cos(2*.9)*np.cos(2*.23)
    source_limit=.7*np.cos(2*.9/np.sqrt(3))*np.cos(2*.23)
    counter_error=max(abs(a.predict([limit_e])[0]-true_limit),abs(b.predict([limit_e])[0]-source_limit))
    assert counter_error<1e-12
    zero_time=max(abs(a.predict([dict(e,time=0)])[0]-sum(np.array(e['weights'])*e['modulations'])*np.cos(e['wavenumber']*e['position'])) for e in all_inputs)
    reversal=max(abs(a.predict([e])[0]-a.predict([dict(e,directions=(-np.array(e['directions'])).tolist(),position=2*np.pi-e['position'])])[0]) for e in all_inputs)
    grid_cases=[hidden[k][4] for k in ['opposed_directions','oblique_directions','three_directions']]
    grid_errors={str(n):max(abs(real_grid_response(e,.27,n)-ref.predict([e])[0]) for e in grid_cases) for n in [64,128,256,512,1024]}
    assert grid_errors['1024']<5e-5 and grid_errors['1024']<grid_errors['64']/100
    # Direct full-intensity weak difference verifies normalization and positivity.
    weak_errors=[];minimum=1
    for e in all_inputs:
        rays=np.array(e['directions']);weights=np.array(e['weights']);mods=np.array(e['modulations']);eps=.01
        profile=weights*(1+eps*mods*np.cos(e['wavenumber']*(e['position']-rays[:,0]*e['time'])))
        minimum=min(minimum,float(min(profile)))
        got=(sum(profile)-sum(weights))*np.exp(-.27*e['time'])/eps
        weak_errors.append(abs(got-ref.predict([e])[0]))
    assert minimum>0 and max(weak_errors)<1e-12 and zero_time<1e-12 and reversal<1e-12
    # Homogeneous positive calibration signals decrease strictly with absorption.
    inputs=ref.calibration_inputs();times=np.array([e['time'] for e in inputs]);amplitude=ref.predict(inputs,0)
    slope_min=float(np.min(times*amplitude*np.exp(-.45*times)))
    assert slope_min>0
    recovery=[];separation=[];anchors=[]
    for truth in np.linspace(.12,.45,13):
        rows=[dict(input=e,value=float(v),sigma=.001) for e,v in zip(inputs,ref.predict(inputs,truth))]
        model=good.Model().fit(rows);recovery.append(abs(model.absorption-truth))
        a.absorption=b.absorption=float(truth)
        separation.append(min(rms(b.predict(es),ref.predict(es,truth)) for k,es in hidden.items() if k!='uniform_and_copropagating'))
        anchors.append(max(abs(a.predict(hidden['uniform_and_copropagating'])-b.predict(hidden['uniform_and_copropagating']))))
    assert max(recovery)<2e-8 and min(separation)>.04 and max(anchors)<1e-12
    return dict(oracle_reference_max=max(errors),jacobian_fd_max=max(fd),zero_flux_exact=True,near_zero_jacobian_error=tiny,
        maximum_imaginary_speed=max(imag),maximum_characteristic_speed=max(speed),opposed_beam_analytic_error=counter_error,
        initial_signal_error=zero_time,spatial_reversal_error=reversal,real_grid_errors=grid_errors,
        weak_signal_normalization_error=max(weak_errors),minimum_tested_finite_intensity=minimum,
        calibration_derivative_magnitude_min=slope_min,noiseless_full_parameter_range_recovery=max(recovery),
        minimum_shortcut_error_over_parameter_range=min(separation),copropagating_anchor_error=max(anchors))


def rms(actual,truth):return float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))


def controls():
    out={}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/radiative_angular_closure_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='radiation-local-') as directory:
            work=Path(directory);shutil.copytree(TASK/'environment',work/'app');shutil.copytree(TASK/'tests',work/'tests');shutil.copy2(path,work/'app/model.py')
            env=dict(os.environ,PYTHONPATH=str(work/'app'),MODEL_PATH=str(work/'app/model.py'),METRICS_PATH=str(work/'metrics.json'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.perf_counter();result=subprocess.run([sys.executable,'-m','pytest','-q',str(work/'app/test_public.py'),str(work/'tests/test_hidden.py')],cwd=work,env=env,capture_output=True,text=True)
            out[label]=dict(returncode=result.returncode,seconds=time.perf_counter()-start,stdout=result.stdout,stderr=result.stderr,metrics=json.loads((work/'metrics.json').read_text()))
            assert result.returncode==(0 if label=='oracle' else 1)
            assert ('9 passed' in result.stdout if label=='oracle' else '3 failed, 6 passed' in result.stdout)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();started=time.perf_counter()
    good=load(TASK/'solution/model.py','oracle');bad=load(ROOT/'scripts/radiative_angular_closure_baseline.py','shortcut');ref=load(TASK/'tests/reference.py','reference')
    metadata=json.loads((TASK/'tests/metadata.json').read_text());sigma=metadata['sigma'];inputs=ref.calibration_inputs();truth=ref.predict(inputs)
    if args.generate:
        observed=truth+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,observed)]
        for name in ['environment','tests']:
            dest=TASK/name/'data/calibration.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();hidden_truth={k:ref.predict(es) for k,es in hidden.items()};diagnostics=[k for k in hidden if k!='uniform_and_copropagating']
    def score(m):return {k:rms(m.predict(es),hidden_truth[k]) for k,es in hidden.items()}
    def chi(m,rows):return float(np.sum(((m.predict(inputs)-np.array([r['value'] for r in rows]))/sigma)**2)/(len(rows)-1))
    report=dict(task='radiative-angular-closure',revision=1,status='staged_unevaluated',calibration_records=len(records),metadata=metadata,controls={})
    for label,module in [('oracle',good),('shortcut',bad)]:
        m=module.Model().fit(records);s=score(m);report['controls'][label]=dict(absorption=m.absorption,calibration_chi2=chi(m,records),hidden=s)
        assert chi(m,records)<1.5 and abs(m.absorption/.27-1)<.03 and s['uniform_and_copropagating']<.04
        assert max(s.values())<.04 if label=='oracle' else min(s[k] for k in diagnostics)>.04
    rng=np.random.default_rng(metadata['noise_validation_seed']);parameters=[];chis=[];oracle_errors=[];shortcut_errors=[]
    for _ in range(256):
        rows=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,truth+rng.normal(0,sigma,len(inputs)))]
        a,b=good.Model().fit(rows),bad.Model().fit(rows);assert abs(a.absorption-b.absorption)<1e-12
        parameters.append(a.absorption);chis.append(chi(a,rows));oracle_errors.append(max(score(a).values()));shortcut_errors.append(min(score(b)[k] for k in diagnostics))
        assert chis[-1]<1.5 and abs(a.absorption/.27-1)<.03 and oracle_errors[-1]<.04 and shortcut_errors[-1]>.04
        assert score(b)['uniform_and_copropagating']<.04
    report['noise']=dict(realizations=256,calibration_passes=256,parameter_passes=256,oracle_passes=256,shortcut_rejections=256,maximum_chi2=max(chis),maximum_relative_parameter_error=float(max(abs(np.array(parameters)/.27-1))),parameter_range=[min(parameters),max(parameters)],oracle_hidden_max=max(oracle_errors),shortcut_hidden_min=min(shortcut_errors))
    report['science']=science(good,bad,ref);report['local_controls']=controls();report['seconds']=time.perf_counter()-started
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]+[Path(__file__),ROOT/'scripts/radiative_angular_closure_baseline.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    path=ROOT/'results/radiative-angular-closure-validation.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
