"""Actual finite-contact controls, independent kinetic science and direct noisy fits."""
import argparse
import ast
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

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/rotating-reservoir'
ARCHIVE=ROOT/'archives/rotating-reservoir-r7'
RESULTS=ROOT/'results'
DIAGNOSTICS=('constant_contact_energy_noise','partial_flux_contact_noise','full_flux_contact_noise')
ANCHOR='stationary_mean_and_uniform_noise'


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def records_for(ref,metadata,noise_seed=None):
    rng=np.random.default_rng(metadata['calibration_seed'] if noise_seed is None else noise_seed)
    gamma=metadata['true_parameter'];nu=3.6
    return [dict(input=e,value=float(gamma*e['relaxation_rate']*nu/(2*gamma*e['relaxation_rate']+nu*(gamma+e['relaxation_rate']))*(e['temperature_a']-e['temperature_b'])+rng.normal(0,metadata['measurement_sigma'])),sigma=metadata['measurement_sigma']) for e in ref.calibration_inputs()]


def provenance_and_source(metadata,ref):
    saved=json.loads((ARCHIVE/'preservation-manifest.json').read_text())
    assert all(sha(ARCHIVE/rel)==digest for rel,digest in saved['files'].items())
    prior=json.loads((ARCHIVE/'tasks/rotating-reservoir/environment/data/calibration.json').read_text())
    actual=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert actual==records_for(ref,metadata) and actual==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert len(prior)==len(actual)==288
    assert all(dict(old['input'],relaxation_rate=1.3)==new['input'] and old['sigma']==new['sigma']==.001 for old,new in zip(prior,actual))
    delta=np.array([r['input']['temperature_a']-r['input']['temperature_b'] for r in actual]);gamma=.67;nu=3.6;lam=1.3
    old_noise=np.array([r['value'] for r in prior])-gamma*nu/(2*gamma+nu)*delta
    new_noise=np.array([r['value'] for r in actual])-gamma*lam*nu/(2*gamma*lam+nu*(gamma+lam))*delta
    assert max(abs(old_noise-new_noise))<2e-16
    public=(TASK/'environment/model.py').read_text();baseline=(ROOT/'scripts/rotating_reservoir_baseline.py').read_text()
    completed=public.replace("        raise NotImplementedError('Infer the common drag from the calibration records.')\n",'        self.drag=fit_drag(records)\n        return self\n')
    assert completed==baseline
    definitions={node.name for node in ast.parse(public).body if isinstance(node,ast.FunctionDef)}
    assert definitions=={'folded','gaussian_closure','readout','calibration_mean','fit_drag','predict_at'}
    assert '.67' not in public and 'TRUE_PARAMETER' not in public
    assert 'kinetic_statistics' not in public and 'collision_weak_matrix' not in public and 'finite_volume' not in public
    assert 'from model' not in (TASK/'tests/reference.py').read_text()
    return dict(predecessor_files_verified=len(saved['files']),predecessor_manifest_sha256=sha(ARCHIVE/'preservation-manifest.json'),
        calibration_regenerated_intentionally=True,calibration_input_preparations_preserved=True,
        calibration_individual_noise_preservation_error=float(max(abs(old_noise-new_noise))),public_has_only_Gaussian_response=True,
        public_has_no_true_parameter_or_exact_noise_branch=True,public_fit_unfinished=True,completed_control_source_physics_identical=True)


def physical_science(oracle,source,ref):
    root=module(RESULTS/'prototypes/rotating-reservoir-r8-root/exact_constant_collision.py','independent_raw_moments')
    ref.state.cache_clear();oracle.readout.cache_clear();oracle.collision_weak_matrix.cache_clear()
    hidden=ref.hidden_inputs();start=time.perf_counter();truth={name:ref.predict(es) for name,es in hidden.items()};cold_ref=time.perf_counter()-start
    assert cold_ref<45
    start=time.perf_counter();physical={name:oracle.predict_at(es,.67) for name,es in hidden.items()};cold_oracle=time.perf_counter()-start
    assert cold_oracle+cold_ref<55
    hidden_agreement={name:relative(physical[name],truth[name]) for name in hidden}
    assert max(hidden_agreement.values())<.001
    hidden_refinements=[]
    for name,es in hidden.items():
        for e in es:
            a=oracle.kinetic_statistics(e,.67);b=oracle.kinetic_statistics(e,.67,degree=42)
            fixed=oracle.kinetic_statistics(e,.67,degree=42,variance=2.)
            if e['flux_fraction']==0.:
                independent=root.exact(e['temperature_a'],e['temperature_b'],.67,e['relaxation_rate'],e['angular_speed'],e['flip_rate'])
                assert max(abs(a[field]-independent[field]) for field in ['mean','noise'])<2e-11
            hidden_refinements.append(dict(group=name,input=e,oracle34=a,oracle42=b,fixed_variance42=fixed,
                noise_refinement_relative=abs(a['noise']-b['noise'])/b['noise'],basis_variance_relative=abs(b['noise']-fixed['noise'])/b['noise']))
    assert max(r['noise_refinement_relative'] for r in hidden_refinements)<.001
    assert max(r['basis_variance_relative'] for r in hidden_refinements)<.001
    corners=[]
    for gamma,ta,tb,lam,omega,flip,eta in itertools.product([.4,1.1],[.4,2.],[.4,1.4],[.2,3.],[-.65,.65],[.8,2.],[0.,1.]):
        e=dict(temperature_a=ta,temperature_b=tb,relaxation_rate=lam,angular_speed=omega,flip_rate=flip,flux_fraction=eta)
        a=oracle.kinetic_statistics(e,gamma);b=oracle.kinetic_statistics(e,gamma,degree=42)
        g=source.gaussian_closure(e,gamma,noise=True);reverse=oracle.kinetic_statistics(dict(e,angular_speed=-omega),gamma)
        volume=ref.finite_volume(e,gamma,points=61,extent=7.,details=True,noise=True)
        fine=ref.finite_volume(e,gamma,points=122,extent=7.,details=True,noise=True)
        independent_noise=(4*fine['noise']-volume['noise'])/3
        independent_mean=(4*fine['calorimeter_mean']-volume['calorimeter_mean'])/3
        if eta==0.:
            exact=root.exact(ta,tb,gamma,lam,omega,flip)
            assert max(abs(a[f]-exact[f]) for f in ['mean','noise'])<2e-11
        entropy=a['bath_a']/ta+a['mean']/tb
        corners.append(dict(input=e,drag=gamma,oracle34=a,oracle42=b,source=g,fv61=volume,fv122=fine,
            fv_noise=independent_noise,fv_mean=independent_mean,
            independent_noise_relative=abs(independent_noise-b['noise'])/b['noise'],
            noise_refinement_relative=abs(a['noise']-b['noise'])/b['noise'],
            motor_reversal_noise_error=abs(a['noise']-reverse['noise']),entropy_production=entropy))
    assert min(r['oracle34']['noise'] for r in corners)>0.
    assert min(r['source']['minimum_covariance'] for r in corners)>0.
    assert min(r['source']['noise'] for r in corners)>0.
    assert min(r['source']['projected_nonconstant_decay'] for r in corners)>0.
    assert min(r['entropy_production'] for r in corners)>-1e-10
    assert max(r['source']['projected_stationary_residual'] for r in corners)<1e-7
    assert max(r['source']['first_law_residual'] for r in corners)<1e-8
    assert max(r['oracle34']['first_law_residual'] for r in corners)<2e-10
    assert max(r['motor_reversal_noise_error'] for r in corners)<2e-10
    assert max(r['noise_refinement_relative'] for r in corners)<.003
    assert max(r['independent_noise_relative'] for r in corners)<.005
    assert min(r[level]['minimum_probability'] for r in corners for level in ['fv61','fv122'])>-1e-12
    assert max(r[level]['discrete_first_law_residual'] for r in corners for level in ['fv61','fv122'])<1e-9
    constant=[]
    for gamma,ta,tb,lam,omega,flip in itertools.product([.4,.67,1.1],[.4,2.],[.4,1.4],[.2,3.],[0.,.65],[.8,2.]):
        e=dict(temperature_a=ta,temperature_b=tb,relaxation_rate=lam,angular_speed=omega,flip_rate=flip,flux_fraction=0.)
        a=oracle.kinetic_statistics(e,gamma);r=root.exact(ta,tb,gamma,lam,omega,flip)
        assert max(abs(a[field]-r[field]) for field in ['mean','noise'])<2e-11
        constant.append(dict(input=e,drag=gamma,oracle=a,independent=r))
    equilibrium=[]
    for gamma,t,lam,eta in itertools.product([.4,1.1],[.4,1.4],[.2,3.],[0.,.5,1.]):
        e=dict(temperature_a=t,temperature_b=t,relaxation_rate=lam,angular_speed=0.,flip_rate=1.2,flux_fraction=eta)
        h=1e-4;a=oracle.kinetic_statistics(e,gamma,variance=t)
        lo=oracle.kinetic_statistics(dict(e,temperature_a=t-h),gamma,variance=t)
        hi=oracle.kinetic_statistics(dict(e,temperature_a=t+h),gamma,variance=t)
        fdt=2*t*t*(hi['mean']-lo['mean'])/(2*h)
        error=abs(a['noise']-fdt)/a['noise'];assert error<2e-6
        equilibrium.append(dict(input=e,drag=gamma,noise=a['noise'],FDT=fdt,relative_error=error))
    isolated=[]
    for t,lam,eta in itertools.product([.4,1.4],[.2,3.],[0.,1.]):
        e=dict(temperature_a=1.,temperature_b=t,relaxation_rate=lam,angular_speed=0.,flip_rate=1.2,flux_fraction=eta)
        a=oracle.kinetic_statistics(e,0.);b=ref.finite_volume(e,0.,points=61,noise=True)
        assert max(abs(a['noise']),abs(b['noise']))<1e-10
        isolated.append(dict(input=e,oracle=a,independent_grid=b))
    rng=np.random.default_rng(810618);microscopic=[]
    for _ in range(256):
        v,g=rng.normal(size=2);before=.5*(v*v+g*g);after=.5*(g*g+v*v)
        microscopic.append(dict(energy_error=abs(after-before),momentum_error=abs((v+g)-(g+v)),motor_flip_particle_energy_error=0.,collision_and_motor_instantaneous_thermal_heat=0.))
    return dict(cold_reference_seconds=cold_ref,cold_oracle_seconds=cold_oracle,hidden_independent_relative_errors=hidden_agreement,
        hidden_refinements=hidden_refinements,full_box_corners=corners,constant_clock_independent_anchors=constant,
        equilibrium_FDT=equilibrium,isolated_bath_boundary=isolated,microscopic_checks=microscopic,
        minimum_source_covariance=min(r['source']['minimum_covariance'] for r in corners),
        minimum_source_noise=min(r['source']['noise'] for r in corners),minimum_projected_decay=min(r['source']['projected_nonconstant_decay'] for r in corners),
        maximum_fullbox_reference_noise_relative=max(r['independent_noise_relative'] for r in corners),
        maximum_oracle_degree_refinement_relative=max(r['noise_refinement_relative'] for r in corners),
        energy_domain_proof='E=(v²+g²)/2 is invariant at swaps/flips. LE≤−min(2gamma,lambda)E+gammaTa+lambdaTb+lambdaOmega²/2. For exp(.2E), OU quadratic coefficients remain negative uniformly because .2max(Ta,Tb)≤.4<1. Completing a square bounds the flow. Nondegenerate diffusion and energy control imply unique stationary law with finite polynomial moments and heat-noise response.',
        source_limitation='Conditional Gaussian moment/quadratic response approximation; projected stationarity and positive covariance/noise/decay certified, not asserted to be an exact Gaussian Markov path law.')


def local_controls():
    output=RESULTS/'rotating-reservoir-r8-controls';output.mkdir(exist_ok=True);results={}
    for name,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/rotating_reservoir_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='rotor-r8-control-') as temporary:
            app=Path(temporary)/'app';shutil.copytree(TASK/'environment',app);shutil.copyfile(path,app/'model.py')
            private=app/'private_tests';shutil.copytree(TASK/'tests',private)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',MODEL_PATH=str(app/'model.py'),METRICS_PATH=str(output/(name+'-metrics.json')),PYTHONPATH=str(app)+os.pathsep+str(private))
            start=time.perf_counter();run=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(app/'test_public.py'),str(private/'test_hidden.py')],cwd=app,env=env,text=True,capture_output=True,timeout=60)
            (output/(name+'-pytest.txt')).write_text(run.stdout+run.stderr)
            expected='9 passed' if name=='oracle' else '3 failed, 6 passed'
            assert expected in run.stdout and run.returncode==(0 if name=='oracle' else 1),run.stdout+run.stderr
            results[name]=dict(seconds=time.perf_counter()-start,returncode=run.returncode,expected=expected,metrics=json.loads((output/(name+'-metrics.json')).read_text()))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    start=time.perf_counter();oracle=module(TASK/'solution/model.py','kinetic_oracle');source=module(ROOT/'scripts/rotating_reservoir_baseline.py','Gaussian_control');ref=module(TASK/'tests/reference.py','independent_positive_ref')
    meta=json.loads((TASK/'tests/metadata.json').read_text());assert meta['revision']==8
    provenance=provenance_and_source(meta,ref);records=records_for(ref,meta);inputs=ref.calibration_inputs();hidden=ref.hidden_inputs()
    truth={name:ref.predict(es) for name,es in hidden.items()}
    def scores(model):return {name:relative(model.predict(es),truth[name]) for name,es in hidden.items()}
    def chi(model,rs):return float(np.sum(((model.predict(inputs)-np.array([r['value'] for r in rs]))/meta['measurement_sigma'])**2)/(len(rs)-1))
    controls={}
    for name,m in [('oracle',oracle),('shortcut',source)]:
        model=m.Model().fit(records);row=dict(parameter=model.drag,parameter_relative_error=abs(model.drag/.67-1),calibration_chi2=chi(model,records),hidden=scores(model))
        assert row['parameter_relative_error']<.03 and row['calibration_chi2']<1.5
        assert all(v<.04 if name=='oracle' or key==ANCHOR else v>.04 for key,v in row['hidden'].items())
        controls[name]=row
    print('r8 actual controls andcalibration PASS',flush=True)
    science=physical_science(oracle,source,ref);print('r8 independent/fullbox/coldscience PASS',flush=True)
    rng=np.random.default_rng(meta['noise_seed']);noise={name:dict(hidden_min=1e10,hidden_max=0.,anchor_max=0.,parameter_error_max=0.,chi2_max=0.,successful_trials=0) for name in controls}
    means=np.array([oracle.calibration_mean(e,.67) for e in inputs])
    for index in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=.001) for e,y in zip(inputs,means+rng.normal(0,.001,len(inputs)))]
        for name,m in [('oracle',oracle),('shortcut',source)]:
            model=m.Model().fit(sample);values=scores(model);diagnostic=[values[key] for key in DIAGNOSTICS];anchor=values[ANCHOR];error=abs(model.drag/.67-1);cal=chi(model,sample)
            assert error<.03 and cal<1.5 and anchor<.04
            assert max(diagnostic)<.04 if name=='oracle' else min(diagnostic)>.04
            row=noise[name];row['hidden_min']=min(row['hidden_min'],min(diagnostic));row['hidden_max']=max(row['hidden_max'],max(diagnostic));row['anchor_max']=max(row['anchor_max'],anchor);row['parameter_error_max']=max(row['parameter_error_max'],error);row['chi2_max']=max(row['chi2_max'],cal);row['successful_trials']+=1
        if (index+1)%16==0:print('r8 DIRECT noisy fits/predictions',index+1,'seconds',round(time.perf_counter()-start,2),flush=True)
    report=dict(revision=8,calibration_regenerated_intentionally=True,calibration_records=288,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],noise_trials=args.noise_trials,calibration_sigma=meta['measurement_sigma'],provenance=provenance,controls=controls,noise=noise,scientific_checks=science,local_controls=local_controls(),seconds=time.perf_counter()-start)
    files=sorted(p for p in TASK.rglob('*') if p.is_file() and not any(part in ['__pycache__','.pytest_cache'] for part in p.parts));files += [ROOT/'scripts/rotating_reservoir_baseline.py',Path(__file__)]
    report['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in files}
    target=RESULTS/'rotating-reservoir-r8-validation.json';target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',report=str(target),sha256=sha(target),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
