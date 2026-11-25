#!/usr/bin/env python3
"""Data generation and independent checks for paired-resonators."""
import argparse
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
TASK=ROOT/'tasks/paired-resonators'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

oracle=load('paired_oracle',TASK/'solution/model.py')
shortcut=load('paired_shortcut',ROOT/'scripts/paired_resonators_baseline.py')
reference=load('paired_reference',TASK/'tests/reference.py')
fock=load('paired_fock',ROOT/'scripts/paired_resonators_fock.py')
metadata=json.loads((TASK/'tests/metadata.json').read_text())

parser=argparse.ArgumentParser()
parser.add_argument('--generate',action='store_true')
parser.add_argument('--samples',type=int,default=256)
parser.add_argument('--skip-fock',action='store_true')
args=parser.parse_args()
calibration=reference.calibration_inputs()*metadata['calibration_repeats']
true_cal=reference.predict(calibration,reference.TRUE_PARAMETER)

def records_for(values):
    return [dict(input=e,value=float(v),sigma=metadata['sigma']) for e,v in zip(calibration,values)]

if args.generate:
    rng=np.random.default_rng(metadata['calibration_seed'])
    records=records_for(true_cal+metadata['sigma']*rng.normal(size=len(calibration)))
    data=json.dumps(records,indent=2)+'\n'
    for relative in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/relative).write_text(data)
records=json.loads((TASK/'tests/data/calibration.json').read_text())
assert records==json.loads((TASK/'environment/data/calibration.json').read_text())
hidden=reference.hidden_inputs()
truth={key:reference.predict(inputs,reference.TRUE_PARAMETER) for key,inputs in hidden.items()}

def error_groups(module,scale):
    return {key:float(np.sqrt(np.mean((module.predict_at(inputs,scale)-truth[key])**2)/np.mean(truth[key]**2))) for key,inputs in hidden.items()}

report={'task':'paired-resonators','revision':1,'calibration_records':len(records),
        'seeds':dict(calibration=metadata['calibration_seed'],noise=metadata['noise_seed'])}
for key,module in [('oracle',oracle),('shortcut',shortcut)]:
    model=module.Model().fit(records)
    residual=(model.predict(calibration)-np.array([r['value'] for r in records]))/metadata['sigma']
    report[key]=dict(energy_scale=model.energy_scale,calibration_chi2=float(residual@residual/(len(records)-1)),hidden=error_groups(module,model.energy_scale))
report['exact_calibration_equivalence']=float(abs(oracle.predict_at(calibration,1.04)-shortcut.predict_at(calibration,1.04)).max())
assert report['exact_calibration_equivalence']==0
assert max(report['oracle']['hidden'].values())<.04
assert all(value>.04 for key,value in report['shortcut']['hidden'].items() if key!='spectroscopy_and_unpaired')
assert report['shortcut']['hidden']['spectroscopy_and_unpaired']<.04

rng=np.random.default_rng(metadata['noise_seed'])
noise=dict(samples=args.samples,calibration_passes=0,parameter_passes=0,oracle_passes=0,shortcut_rejections=0,
           max_chi2=0.,max_parameter_error=0.,oracle_max_error=0.,shortcut_min_discriminating_error=float('inf'),max_fit_difference=0.)
for _ in range(args.samples):
    noisy=records_for(true_cal+metadata['sigma']*rng.normal(size=len(calibration)))
    good=oracle.Model().fit(noisy);bad=shortcut.Model().fit(noisy)
    residual=(good.predict(calibration)-np.array([r['value'] for r in noisy]))/metadata['sigma']
    chi=float(residual@residual/(len(records)-1));pe=abs(good.energy_scale/reference.TRUE_PARAMETER-1)
    ge=error_groups(oracle,good.energy_scale);be=error_groups(shortcut,bad.energy_scale)
    noise['calibration_passes']+=chi<1.5;noise['parameter_passes']+=pe<.03
    noise['oracle_passes']+=max(ge.values())<.04
    noise['shortcut_rejections']+=all(v>.04 for k,v in be.items() if k!='spectroscopy_and_unpaired')
    noise['max_chi2']=max(noise['max_chi2'],chi);noise['max_parameter_error']=max(noise['max_parameter_error'],pe)
    noise['oracle_max_error']=max(noise['oracle_max_error'],max(ge.values()))
    noise['shortcut_min_discriminating_error']=min(noise['shortcut_min_discriminating_error'],min(v for k,v in be.items() if k!='spectroscopy_and_unpaired'))
    noise['max_fit_difference']=max(noise['max_fit_difference'],abs(good.energy_scale-bad.energy_scale))
for key in ['calibration_passes','parameter_passes','oracle_passes','shortcut_rejections']:assert noise[key]==args.samples,(key,noise)
report['noise']={key:int(value) if isinstance(value,np.integer) else value for key,value in noise.items()}

sigma=np.diag([1.,1.,1.,-1.,-1.,-1.])
checks=dict(reference_max_abs=0.,reference_max_relative=0.,minimum_quadratic_eigenvalue=float('inf'),
            minimum_frequency=float('inf'),minimum_signature_norm=float('inf'),canonical_defect=0.,diagonalization_error=0.,
            unpaired_equivalence=0.,ground_output_error=0.,minimum_thermal_excess=float('inf'),classical_slope_error=0.)
rng=np.random.default_rng(870631)
settings=[(s,t,scale) for s in [0.,.2,.6,1.] for t in [.0,.15,.8] for scale in [.8,1.04,1.3]]
settings += [(rng.uniform(0,1),rng.uniform(.15,.8),rng.uniform(.8,1.3)) for _ in range(256)]
for s,t,scale in settings:
    h=np.block([[oracle.A,s*oracle.B],[s*oracle.B,oracle.A]])
    eigen,vectors=np.linalg.eig(sigma@h);chosen=np.argsort(eigen.real)[3:]
    energies=eigen[chosen].real;v=vectors[:,chosen]
    norm=np.einsum('ij,ij->j',v.conj(),sigma@v).real
    v=v/np.sqrt(norm);u,z=v[:3],v[3:]
    transform=np.block([[u,z.conj()],[z,u.conj()]])
    checks['minimum_quadratic_eigenvalue']=min(checks['minimum_quadratic_eigenvalue'],float(np.linalg.eigvalsh(h).min()))
    checks['minimum_frequency']=min(checks['minimum_frequency'],float(energies.min()))
    checks['minimum_signature_norm']=min(checks['minimum_signature_norm'],float(norm.min()))
    checks['canonical_defect']=max(checks['canonical_defect'],float(abs(transform.conj().T@sigma@transform-sigma).max()))
    checks['diagonalization_error']=max(checks['diagonalization_error'],float(abs(transform.conj().T@h@transform-np.diag(np.r_[energies,energies])).max()))
    experiments=[reference.experiment(readout,j,s,t) for readout in ['frequency','thermal_excess'] for j in range(3)]
    good=oracle.predict_at(experiments,scale);ref=reference.predict(experiments,scale)
    checks['reference_max_abs']=max(checks['reference_max_abs'],float(abs(good-ref).max()))
    mask=abs(ref)>1e-14
    checks['reference_max_relative']=max(checks['reference_max_relative'],float(abs((good[mask]-ref[mask])/ref[mask]).max()))
    checks['minimum_thermal_excess']=min(checks['minimum_thermal_excess'],float(good[3:].min()))
    if s==0:
        checks['unpaired_equivalence']=max(checks['unpaired_equivalence'],float(abs(good-shortcut.predict_at(experiments,scale)).max()))
    if t==0:checks['ground_output_error']=max(checks['ground_output_error'],float(abs(good[3:]).max()))
for s in [0.,.5,1.]:
    inputs=[reference.experiment(index=j,pairing=s,temperature=1e4) for j in range(3)]
    inputs2=[dict(e,temperature=2e4) for e in inputs]
    slope=(oracle.predict_at(inputs2,1.04)-oracle.predict_at(inputs,1.04))/1e4
    exact=.5/1.04*np.diag(np.linalg.inv(oracle.A+s*oracle.B)+np.linalg.inv(oracle.A-s*oracle.B))
    checks['classical_slope_error']=max(checks['classical_slope_error'],float(abs(slope-exact).max()))
assert checks['reference_max_relative']<1e-11
assert checks['canonical_defect']<1e-12 and checks['diagonalization_error']<1e-12
assert checks['minimum_quadratic_eigenvalue']>0 and checks['minimum_frequency']>0 and checks['minimum_signature_norm']>0
assert checks['minimum_thermal_excess']>=0 and checks['ground_output_error']==0
assert checks['classical_slope_error']<2e-8
report['physical_checks']=checks

fits=[]
for scale in [.8,1.04,1.3]:
    fitted=oracle.Model().fit(records_for(reference.predict(calibration,scale))).energy_scale
    assert abs(fitted-scale)<1e-12
    fits.append(dict(true=scale,fit=fitted))
report['noiseless_fit_recovery']=fits
report['fit_quadratic_curvature']=float(2*np.sum((oracle.predict_at(calibration,1.)/metadata['sigma'])**2))

if not args.skip_fock:
    fock_report=[]
    for pairing,cutoffs in [(.6,[18,30]),(1.,[30,44,60])]:
        temperatures=[.25,.45]
        exact=np.array([[reference.predict([reference.experiment(index=j,pairing=pairing,temperature=t)],1.04)[0] for j in range(3)] for t in temperatures])
        previous=None
        for cutoff in cutoffs:
            start=time.monotonic()
            values,last_gap,count=fock.fock(cutoff,pairing,temperatures,scale=1.04,levels=120)
            values=np.array(values)
            row=dict(pairing=pairing,cutoff=cutoff,temperatures=temperatures,energy_scale=1.04,
                     maximum_error=float(abs(values-exact).max()),max_change=None if previous is None else float(abs(values-previous).max()),
                     highest_included_gap=float(last_gap),retained_states=count,seconds=time.monotonic()-start)
            fock_report.append(row);previous=values
            print('Fock check',row,flush=True)
        assert fock_report[-1]['maximum_error']<2e-8
    # Retained eigenstate count is varied independently of the Fock basis cutoff.
    values,_,_=fock.fock(60,1.,[.25,.45],scale=1.04,levels=180)
    level_change=float(abs(np.array(values)-previous).max())
    assert level_change<2e-10
    report['finite_fock_checks']=dict(cutoff_refinement=fock_report,eigenstate_count_120_to180_change=level_change,
      scope='Independent full3-mode Fock check at moderate and maximal pairing, T=.25,.45. The exact real-quadrature reference covers the full public temperature/scale domain.')

def local_control(source):
    with tempfile.TemporaryDirectory(prefix='paired-control-') as directory:
        directory=Path(directory);shutil.copy2(source,directory/'model.py')
        shutil.copy2(TASK/'environment/test_public.py',directory/'test_public.py');shutil.copytree(TASK/'environment/data',directory/'data')
        env=dict(os.environ,PYTHONPATH=str(directory),OPENBLAS_NUM_THREADS='1')
        start=time.monotonic();p=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(directory/'test_public.py'),str(TASK/'tests/test_hidden.py')],capture_output=True,text=True,env=env)
        return dict(returncode=p.returncode,seconds=time.monotonic()-start,stdout=p.stdout,stderr=p.stderr)
report['local_controls']={'oracle':local_control(TASK/'solution/model.py'),'shortcut':local_control(ROOT/'scripts/paired_resonators_baseline.py')}
assert report['local_controls']['oracle']['returncode']==0
assert '3 failed, 5 passed' in report['local_controls']['shortcut']['stdout']
path=ROOT/'results/paired-resonators-validation.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
