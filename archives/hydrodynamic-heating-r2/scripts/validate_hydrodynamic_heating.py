"""Validate the staged viscous-fluid local calorimeter and both completed controls."""
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

STAGE = Path(__file__).resolve().parents[1]
TASK = STAGE/'tasks/hydrodynamic-heating'
RESULTS = STAGE/'results'


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nrmse(actual, truth):
    return float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))


def local_control(path, label):
    with tempfile.TemporaryDirectory(prefix='viscous-local-') as directory:
        root=Path(directory);shutil.copytree(TASK/'environment',root/'app');shutil.copytree(TASK/'tests',root/'tests')
        shutil.copy2(path,root/'app/model.py')
        metrics=root/'metrics.json'
        environment=dict(os.environ,MODEL_PATH=str(root/'app/model.py'),METRICS_PATH=str(metrics),
                         PYTHONPATH=str(root/'app'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        start=time.monotonic()
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(root/'app/test_public.py'),str(root/'tests/test_hidden.py')],
                           cwd=root/'app',env=environment,text=True,capture_output=True,timeout=60)
        return dict(returncode=run.returncode,seconds=time.monotonic()-start,stdout=run.stdout,stderr=run.stderr,
                    metrics=json.loads(metrics.read_text()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--regenerate',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    oracle=load(TASK/'solution/model.py','oracle');shortcut=load(STAGE/'scripts/hydrodynamic_heating_baseline.py','shortcut')
    reference=load(TASK/'tests/reference.py','reference');meta=json.loads((TASK/'tests/metadata.json').read_text())
    p=reference.TRUE_PARAMETER;inputs=reference.calibration_inputs();clean=oracle.Model();clean.plasma_frequency=p
    mean=clean.predict(inputs)
    if args.regenerate:
        noise=np.random.default_rng(meta['calibration_seed']).normal(0,meta['sigma'],len(inputs))
        records=[dict(input=e,value=float(y),sigma=meta['sigma']) for e,y in zip(inputs,mean+noise)]
        text=json.dumps(records,indent=2)+'\n'
        for part in ['environment','tests']:(TASK/part/'data/calibration.json').write_text(text)
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    groups=reference.hidden_inputs();truth={name:reference.predict(es) for name,es in groups.items()}
    controls={}
    for name,module in [('oracle',oracle),('shortcut',shortcut)]:
        model=module.Model().fit(records);fit=model.predict(inputs)
        chi=float(np.mean(((fit-np.array([r['value'] for r in records]))/meta['sigma'])**2)*len(records)/(len(records)-1))
        controls[name]=dict(parameter=model.plasma_frequency,calibration_chi2=chi,
            hidden={key:nrmse(model.predict(es),truth[key]) for key,es in groups.items()})
    print('Nominal controls',json.dumps(controls),flush=True)
    assert all(v['calibration_chi2']<1.5 and abs(v['parameter']/p-1)<.03 for v in controls.values())
    assert max(controls['oracle']['hidden'].values())<.04
    assert all(controls['shortcut']['hidden'][key]>.04 for key in ['front_window','interior_window','rear_window'])
    assert controls['shortcut']['hidden']['whole_slab']<.04
    rng=np.random.default_rng(meta['noise_seed']);noise_metrics=[]
    noise_hidden={'oracle':[], 'shortcut':[]};fit_equivalence=[]
    for iteration in range(256):
        noisy=[dict(r,value=float(y)) for r,y in zip(records,mean+rng.normal(0,meta['sigma'],len(mean)))]
        model=oracle.Model().fit(noisy);ps=model.plasma_frequency
        chi=float(np.sum(((model.predict(inputs)-[r['value'] for r in noisy])/meta['sigma'])**2)/(len(inputs)-1))
        other=shortcut.Model().fit(noisy)
        fit_equivalence.append(abs(other.plasma_frequency-ps))
        for label,control in [('oracle',model),('shortcut',other)]:
            noise_hidden[label].append({key:nrmse(control.predict(es),truth[key]) for key,es in groups.items()})
        noise_metrics.append([ps,chi])
    noise_metrics=np.array(noise_metrics)
    assert max(fit_equivalence)<1e-8
    assert max(value for row in noise_hidden['oracle'] for value in row.values())<.04
    assert min(row[key] for row in noise_hidden['shortcut'] for key in ['front_window','interior_window','rear_window'])>.04
    assert max(row['whole_slab'] for row in noise_hidden['shortcut'])<.04
    extrema={} 
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        scores=[]
        for parameter in [float(noise_metrics[:,0].min()),float(noise_metrics[:,0].max())]:
            model=module.Model();model.plasma_frequency=parameter
            scores.append({key:nrmse(model.predict(es),truth[key]) for key,es in groups.items()})
        extrema[label]=scores
    assert np.all(noise_metrics[:,1]<1.5) and np.max(abs(noise_metrics[:,0]/p-1))<.03
    assert max(v for row in extrema['oracle'] for v in row.values())<.04
    assert min(row[key] for row in extrema['shortcut'] for key in ['front_window','interior_window','rear_window'])>.04
    science={};cases=[]
    for pp in [.85,1.15]:
        for w in [.7,1.5]:
            for d in [.3,1.2]:
                for a in [0.,1.]:cases.append((dict(frequency=w,thickness=d,angle=a,window=[0.,1.]),pp))
    rng=np.random.default_rng(110723)
    for _ in range(32):
        left=rng.uniform(0,.75);right=rng.uniform(left+.15,1.)
        cases.append((dict(frequency=float(rng.uniform(.7,1.5)),thickness=float(rng.uniform(.3,1.2)),
                          angle=float(rng.uniform(-1,1)),window=[left,right]),float(rng.uniform(.85,1.15))))
    errors=[];refinements=[];balances=[];cal_diffs=[];flux_errors=[];bc_errors=[];strain_errors=[];parity=[];quad=[];source_values=[];physical_values=[]
    for e,pp in cases:
        exact=oracle.response(e,pp);approx=shortcut.response(e,pp)
        independent=reference.response(e,pp)
        errors.append(abs(exact-independent));source_values.append(approx);physical_values.append(exact)
        refinements.append(abs(independent-reference.response(e,pp,tolerance=1e-9,order=144)))
        w,d,a=(e[k] for k in ['frequency','thickness','angle']);c=oracle.amplitudes(w,d,a,pp)
        ends=oracle.mode_matrix([0,d],w,d,a,pp)@c
        whole=dict(e,window=[0.,1.]);full=oracle.response(whole,pp)
        balances.append(abs(full-(1-abs(ends[0,0]-1)**2-abs(ends[1,0])**2)))
        cal_diffs.append(abs(full-shortcut.response(whole,pp)))
        normal=w*np.cos(a)
        bc_errors.extend([abs(ends[0,1]+1j*normal*ends[0,0]-2j*normal),abs(ends[1,1]-1j*normal*ends[1,0]),abs(ends[0,2]),abs(ends[1,2])])
        depth=np.array(e['window'])*d
        jends=oracle.mode_matrix(depth,w,d,a,pp)@c
        boundary=oracle.VISCOSITY*np.diff(np.real(np.conj(jends[:,2])*jends[:,3]))[0]/(pp*pp*np.cos(a))
        flux_errors.append(abs(exact-approx-boundary))
        x,weights=np.polynomial.legendre.leggauss(128);lo,hi=depth;z=(lo+hi)/2+(hi-lo)*x/2
        E,Ez,J,Jz=(oracle.mode_matrix(z,w,d,a,pp)@c).T;k=w*np.sin(a)
        grad=np.zeros((len(z),3,3),complex);grad[:,1,0]=1j*k*J;grad[:,1,2]=Jz
        symmetric=grad+grad.transpose(0,2,1)
        contraction=np.sum(abs(symmetric)**2,axis=(1,2))/2
        strain_errors.append(float(np.max(abs(contraction-(abs(Jz)**2+k*k*abs(J)**2)))))
        heat=oracle.GAMMA*abs(J)**2+oracle.VISCOSITY*contraction
        quad.append(abs(exact-(hi-lo)/2*np.dot(weights,heat)/(pp*pp*np.cos(a))))
        parity.append(abs(exact-oracle.response(dict(e,angle=-a),pp)))
    science.update(independent_cases=len(cases),oracle_reference_max=max(errors),reference_refinement_max=max(refinements),
        optical_energy_balance_max=max(balances),full_slab_closure_equivalence_max=max(cal_diffs),
        window_stress_flux_identity_max=max(flux_errors),boundary_residual_max=max(bc_errors),
        symmetric_strain_contraction_max=max(strain_errors),quadrature64_to128_max=max(quad),
        angle_reversal_max=max(parity),minimum_tested_correct_heat=min(physical_values),minimum_tested_shortcut_heat=min(source_values))
    assert max(errors)<1e-8 and max(refinements)<1e-8
    assert max(balances)<1e-10 and max(cal_diffs)<1e-10 and max(flux_errors)<1e-10
    assert max(bc_errors)<1e-10 and max(strain_errors)<1e-10 and max(quad)<1e-10 and max(parity)<1e-10
    # The calibration uses active viscosity and finite slabs; check full-range fitting.
    recovery=[];profile_valid=[]
    unique=inputs[:16];grid=np.linspace(.85,1.15,121)
    table=np.array([[oracle.response(e,pp) for e in unique] for pp in grid])
    for pp in np.linspace(.85,1.15,13):
        noiseless=[dict(input=e,value=oracle.response(e,pp),sigma=meta['sigma']) for e in unique]
        fit=oracle.Model().fit(noiseless).plasma_frequency;recovery.append(abs(fit-pp))
        target=np.array([r['value'] for r in noiseless]);profile=np.sum((table-target)**2,axis=1);minimum=int(np.argmin(profile))
        profile_valid.append(bool(np.all(np.diff(profile[:minimum+1])<0) and np.all(np.diff(profile[minimum:])>0)))
    science['full_parameter_fit_error_max']=max(recovery);science['unimodal_parameter_profiles']=all(profile_valid)
    assert max(recovery)<1e-7 and all(profile_valid)
    local={'oracle':local_control(TASK/'solution/model.py','oracle'),
           'shortcut':local_control(STAGE/'scripts/hydrodynamic_heating_baseline.py','shortcut')}
    assert local['oracle']['returncode']==0
    assert local['shortcut']['returncode']==1 and '3 failed' in local['shortcut']['stdout']
    report=dict(task='hydrodynamic-heating',revision=2,status='scientifically_validated_staged_unevaluated',
        calibration_records=len(records),metadata=meta,controls=controls,
        noise=dict(realizations=256,calibration_passes=256,parameter_passes=256,
                   oracle_passes=256,shortcut_rejections=256,maximum_control_fit_difference=max(fit_equivalence),
                   all_sample_hidden_ranges={label:{key:[min(row[key] for row in rows),max(row[key] for row in rows)] for key in groups} for label,rows in noise_hidden.items()},
                   minimum_parameter=float(noise_metrics[:,0].min()),maximum_parameter=float(noise_metrics[:,0].max()),
                   maximum_chi2=float(noise_metrics[:,1].max()),maximum_relative_parameter_error=float(np.max(abs(noise_metrics[:,0]/p-1))),
                   extrema_hidden=extrema),science=science,local_controls=local,seconds=time.monotonic()-start)
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in str(p)]+list((STAGE/'scripts').glob('*.py'))
    report['source_sha256']={str(p.relative_to(STAGE)):sha(p) for p in sorted(files)}
    RESULTS.mkdir(exist_ok=True);path=RESULTS/'hydrodynamic-heating-r2-validation.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',path,json.dumps(science),flush=True)

if __name__=='__main__':main()
