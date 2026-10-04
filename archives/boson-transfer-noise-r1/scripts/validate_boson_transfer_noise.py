"""Scientific controls for bosonic two-time energy-transfer variance."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/boson-transfer-noise'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


good=load(TASK/'solution/model.py','good')
bad=load(ROOT/'scripts/boson_transfer_noise_baseline.py','bad')
ref=load(TASK/'tests/reference.py','reference')


def fock_transfer(transmission,left,right,maximum_total):
    """Two projective number measurements around a beamsplitter unitary.

    Sum exact fixed-total-number blocks; no Gaussian moment or transmission
    noise formula is used here. Return retained probability, mean and variance.
    """
    angle=np.arcsin(np.sqrt(transmission))
    ratio_left=left/(1+left)
    ratio_right=right/(1+right)
    probability=mean=second=0.
    for total in range(maximum_total+1):
        occupation=np.arange(total+1)
        generator=np.zeros((total+1,total+1))
        ladder=np.sqrt((occupation[:-1]+1)*(total-occupation[:-1]))
        generator[occupation[:-1]+1,occupation[:-1]]=ladder
        generator[occupation[:-1],occupation[:-1]+1]=-ladder
        unitary=expm(angle*generator)
        initial=(ratio_left**occupation)*(ratio_right**(total-occupation))/((1+left)*(1+right))
        transition=abs(unitary)**2*initial[None,:]
        # Basis label is left occupation; gain on right is initial_left-final_left.
        change=occupation[None,:]-occupation[:,None]
        probability+=float(transition.sum())
        mean+=float(np.sum(transition*change))
        second+=float(np.sum(transition*change**2))
    return np.array([probability,mean,second-mean**2])


def run(count):
    records=json.loads((TASK/'tests/data/calibration.json').read_text())
    inputs=[r['input'] for r in records]
    sigma=np.array([r['sigma'] for r in records])
    clean=ref.predict(inputs,.55)
    hidden=ref.hidden_inputs()
    truth={key:ref.predict(es,.55) for key,es in hidden.items()}
    def scores(module,g):
        return {key:float(np.sqrt(np.mean((module.predict_at(es,g)-truth[key])**2)/np.mean(truth[key]**2))) for key,es in hidden.items()}
    controls={}
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records)
        residual=(model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        controls[label]=dict(coupling=model.coupling,calibration_chi2=float(residual@residual/(len(records)-1)),hidden=scores(module,model.coupling))
    samples=[]
    rng=np.random.default_rng(735193)
    for _ in range(count):
        values=clean+sigma*rng.normal(size=len(inputs))
        noisy=[dict(input=e,value=float(v),sigma=float(s)) for e,v,s in zip(inputs,values,sigma)]
        a=good.Model().fit(noisy)
        b=bad.Model().fit(noisy)
        assert a.coupling==b.coupling
        residual=(a.predict(inputs)-values)/sigma
        samples.append(dict(coupling=a.coupling,chi2=float(residual@residual/(len(inputs)-1)),oracle=scores(good,a.coupling),shortcut=scores(bad,b.coupling)))
    all_inputs=inputs+sum(hidden.values(),[])
    exact_cal=reference_error=refinement=unitarity=equilibrium_fdt=reversal=current_conservation=zero_transmission=0.
    transmission_range=[1.,0.]
    noiseless_recovery=[]
    profile=[]
    for g in [.3,.4,.55,.65,.8]:
        predicted=good.predict_at(all_inputs,g)
        reference_error=max(reference_error,float(max(abs(predicted-ref.predict(all_inputs,g)))))
        refinement=max(refinement,float(max(abs(predicted-good.predict_at(all_inputs,g,160)))))
        exact_cal=max(exact_cal,float(max(abs(good.predict_at(inputs,g)-bad.predict_at(inputs,g)))))
        ideal=good.predict_at(inputs,g)
        fitted=good.Model().fit([dict(input=e,value=float(v),sigma=.0005) for e,v in zip(inputs,ideal)])
        noiseless_recovery.append(dict(true=g,fitted=fitted.coupling))
        for shift in [-.2,.2]:
            for omega in np.linspace(.4,3.4,81):
                s=ref.scattering(omega,shift,g)
                unitarity=max(unitarity,float(np.max(abs(s.conj().T@s-np.eye(2)))))
                transmission=float(abs(s[1,0])**2)
                transmission_range=[min(transmission_range[0],transmission),max(transmission_range[1],transmission)]
                changes=[s.conj().T@np.diag(p)@s-np.diag(p) for p in [[1.,0.],[0.,1.]]]
                current_conservation=max(current_conservation,float(np.max(abs(sum(changes)))))
            temperature=1.2
            eq=ref.experiment(shift,temperature,temperature,[.4,3.4])
            delta=1e-4
            plus=dict(eq,left_temperature=temperature+delta/2,right_temperature=temperature-delta/2,readout='current')
            minus=dict(eq,left_temperature=temperature-delta/2,right_temperature=temperature+delta/2,readout='current')
            conductance=(good.predict_at([plus],g)[0]-good.predict_at([minus],g)[0])/(2*delta)
            equilibrium_fdt=max(equilibrium_fdt,abs(good.predict_at([eq],g)[0]-2*temperature**2*conductance))
            es=[ref.experiment(shift,3.,.15,[.4,3.4],readout) for readout in ['current','noise']]
            swapped=[dict(e,left_temperature=e['right_temperature'],right_temperature=e['left_temperature']) for e in es]
            reversal=max(reversal,float(max(abs(good.predict_at(es,g)-np.array([-1.,1.])*good.predict_at(swapped,g)))))
            zero_transmission=max(zero_transmission,float(max(abs(good.predict_at(es,0.)))))
    for g in np.linspace(.3,.8,101):
        residual=(good.predict_at(inputs,g)-clean)/sigma
        profile.append([float(g),float(residual@residual)])
    minima=[i for i in range(1,len(profile)-1) if profile[i][1]<profile[i-1][1] and profile[i][1]<profile[i+1][1]]
    corners=[ref.experiment(s,left,right,band,readout) for s in [-.2,.2] for left,right in [(.15,.15),(3.,3.),(.15,3.),(3.,.15)] for band in [[.4,.6],[.4,3.4],[3.2,3.4]] for readout in ['current','noise']]
    minimum_mode_frequency=min(float(np.linalg.eigvalsh([[1.4+s,g],[g,1.9+s]])[0]) for s in [-.2,.2] for g in [.3,.8])
    minimum_noise=min(float(good.predict_at([e],g)[0]) for e in corners if e['readout']=='noise' for g in [.3,.8])
    corner_error=max(float(max(abs(good.predict_at(corners,g)-ref.predict(corners,g)))) for g in [.3,.8])
    corner_refinement=max(float(max(abs(good.predict_at(corners,g)-good.predict_at(corners,g,160)))) for g in [.3,.8])
    fock=[]
    for transmission,left,right in [(.15,.7,.2),(.5,1.,.1),(.85,.4,.9),(1.,.6,.6),(0.,1.,.2)]:
        coarse=fock_transfer(transmission,left,right,48)
        fine=fock_transfer(transmission,left,right,64)
        # Compare direct finite-Fock TPM moments with independent operator contractions.
        s=np.array([[np.sqrt(1-transmission),-np.sqrt(transmission)],[np.sqrt(transmission),np.sqrt(1-transmission)]])
        a=s.T@np.diag([0.,1.])@s-np.diag([0.,1.]);n=np.array([left,right])
        expected=np.array([1.,np.sum(np.diag(a)*n),sum(a[i,j]*a[j,i]*n[i]*(1+n[j]) for i in range(2) for j in range(2))])
        fock.append(dict(transmission=transmission,left_occupation=left,right_occupation=right,error=float(max(abs(fine-expected))),refinement=float(max(abs(coarse-fine))),retained_probability=float(fine[0])))
    # Independent-event closure is asymptotically correct to first order in transmission.
    weak=[]
    es=[ref.experiment(.1,2.,.3,[2.8,3.4])]
    for g in [.01,.005,.0025]:
        difference=good.predict_at(es,g)[0]-bad.predict_at(es,g)[0]
        weak.append([g,float(difference),float(difference/g**4)])
    local={}
    for label,source in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/boson_transfer_noise_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='boson-transfer-'+label+'-') as directory:
            base=Path(directory);shutil.copytree(TASK/'environment',base/'app');shutil.copytree(TASK/'tests',base/'tests');shutil.copy2(source,base/'app/model.py')
            p=subprocess.run([sys.executable,'-m','pytest','-q',str(base/'app/test_public.py'),str(base/'tests/test_hidden.py')],cwd=base/'app',env=dict(os.environ,PYTHONPATH=str(base/'app'),OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
            local[label]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    report=dict(revision=1,noise_trials=count,calibration_seed=735191,noise_seed=735193,controls=controls,calibration_equivalence=exact_cal,all_scored_reference_error=reference_error,quadrature80_to160_error=refinement,corner_reference_error=corner_error,corner_refinement=corner_refinement,minimum_hamiltonian_frequency=minimum_mode_frequency,minimum_corner_noise=minimum_noise,scattering_unitarity_error=unitarity,transmission_range=transmission_range,operator_energy_conservation_error=current_conservation,temperature_swap_error=reversal,zero_coupling_error=zero_transmission,equilibrium_fdt_error=equilibrium_fdt,finite_fock_checks=fock,weak_transmission_difference=weak,noiseless_recovery=noiseless_recovery,calibration_profile=profile,calibration_profile_local_minima=minima,
        noise=dict(calibration_passes=sum(s['chi2']<1.5 for s in samples),parameter_passes=sum(abs(s['coupling']/.55-1)<.03 for s in samples),oracle_passes=sum(max(s['oracle'].values())<.025 for s in samples),shortcut_passes=sum(max(s['shortcut'].values())<.025 for s in samples),maximum_oracle_error=max(max(s['oracle'].values()) for s in samples),minimum_shortcut_error=min(min(s['shortcut'].values()) for s in samples),maximum_chi2=max(s['chi2'] for s in samples),coupling_range=[min(s['coupling'] for s in samples),max(s['coupling'] for s in samples)]),local_pytest=local,noise_realizations=samples)
    assert exact_cal<1e-12 and reference_error<1e-8 and refinement<1e-8 and corner_error<1e-8 and corner_refinement<1e-8
    assert unitarity<1e-12 and current_conservation<1e-12 and reversal<1e-12 and zero_transmission<1e-12 and equilibrium_fdt<1e-8
    assert minimum_mode_frequency>0 and minimum_noise>=0
    assert abs(weak[-1][2]/weak[0][2]-1)<.001
    assert 0<=transmission_range[0]<=transmission_range[1]<=1+1e-12
    assert max(x['error'] for x in fock)<1e-10 and max(x['refinement'] for x in fock)<1e-9
    assert len(minima)==1 and abs(profile[minima[0]][0]-.55)<1e-12
    assert max(abs(x['fitted']-x['true']) for x in noiseless_recovery)<1e-7
    assert all(report['noise'][k]==count for k in ['calibration_passes','parameter_passes','oracle_passes'])
    assert report['noise']['shortcut_passes']==0 and report['noise']['minimum_shortcut_error']>.025
    assert local['oracle']['returncode']==0 and local['shortcut']['returncode']==1
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    if args.generate:
        inputs=ref.calibration_inputs();clean=ref.predict(inputs,.55);rng=np.random.default_rng(735191)
        records=[dict(input=e,value=float(v+.0005*rng.normal()),sigma=.0005) for e,v in zip(inputs,clean)]
        for part in ['environment','tests']:(TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    report=run(args.noise_trials)
    (ROOT/'results/boson-transfer-noise-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['noise_realizations','local_pytest','calibration_profile']},indent=2))
