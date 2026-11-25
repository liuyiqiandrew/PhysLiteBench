"""Noise robustness, Gaussian/Fock references and local verifier controls."""
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
from scipy.linalg import eigh_tridiagonal
from scipy.special import zeta

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/bogoliubov-momentum'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


good=load(TASK/'solution/model.py','oracle')
bad=load(ROOT/'scripts/bogoliubov_momentum_baseline.py','shortcut')
ref=load(TASK/'tests/reference.py','reference')


def fock_pair(kinetic,interaction,temperature,cutoff):
    energies=[];differences=[];degeneracies=[]
    for difference in range(cutoff):
        n=np.arange(cutoff-difference)
        diagonal=(kinetic+interaction)*(2*n+difference)
        off_diagonal=interaction*np.sqrt((n[:-1]+1)*(n[:-1]+difference+1))
        energies.extend(eigh_tridiagonal(diagonal,off_diagonal,eigvals_only=True))
        differences.extend(np.full(len(n),difference**2));degeneracies.extend(np.full(len(n),1 if difference==0 else 2))
    energies=np.array(energies);weights=np.array(degeneracies)*np.exp(-(energies-energies.min())/temperature)
    return float(weights@differences/weights.sum())


def validate(count):
    records=json.loads((TASK/'tests/data/calibration.json').read_text());inputs=[r['input'] for r in records];sigma=np.array([r['sigma'] for r in records]);clean=ref.predict(inputs,1.08)
    hidden=ref.hidden_inputs();truth={key:ref.predict(es,1.08) for key,es in hidden.items()}
    def scores(module,gain):return {key:float(np.sqrt(np.mean((module.predict_at(es,gain)-truth[key])**2)/np.mean(truth[key]**2))) for key,es in hidden.items()}
    controls={}
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records);res=(model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        controls[label]=dict(variance_gain=model.variance_gain,calibration_chi2=float(res@res/(len(records)-1)),hidden=scores(module,model.variance_gain))
    rng=np.random.default_rng(456833);samples=[]
    for _ in range(count):
        observed=clean+sigma*rng.normal(size=len(inputs));noisy=[dict(input=e,value=float(v),sigma=float(s)) for e,v,s in zip(inputs,observed,sigma)]
        a=good.Model().fit(noisy);b=bad.Model().fit(noisy);assert abs(a.variance_gain-b.variance_gain)<1e-10
        res=(a.predict(inputs)-observed)/sigma
        samples.append(dict(variance_gain=a.variance_gain,chi2=float(res@res/(len(inputs)-1)),oracle=scores(good,a.variance_gain),shortcut=scores(bad,b.variance_gain)))
    grid=[ref.experiment(t,u) for t in [0.,.08,.12,.25,.5,.9] for u in [0.,.01,.1,.4,.7,1.]]
    reference_error=reference_relative=refinement=ground_depletion=ideal_variance=ideal_equivalence=0.
    for e in grid:
        t=e['temperature'];u=e['interaction'];actual=good.variance_density(t,u);reference=ref.unit_variance(t,u)
        reference_error=max(reference_error,abs(actual-reference))
        if reference>0:reference_relative=max(reference_relative,abs(actual/reference-1))
        refinement=max(refinement,abs(actual-good.variance_density(t,u,320)))
        ground_depletion=max(ground_depletion,abs(good.depletion(0.,u)-u**1.5/(3*np.pi**2)))
        if u==0:
            expected=t*(t/(2*np.pi))**1.5*zeta(1.5,1.)
            ideal_variance=max(ideal_variance,abs(actual-expected));ideal_equivalence=max(ideal_equivalence,abs(actual-bad.variance_density(t,u)))
    fock=[]
    for e,u,t in [(.25,.6,.3),(.08,.4,.1),(.5,1.,.5),(.3,0.,.6)]:
        na,an=ref.atomic_covariances(np.sqrt(2*e),t,u);wick=2*(na*(na+1)-an*an)
        values={str(n):fock_pair(e,u,t,n) for n in [24,48,80,120]}
        fock.append(dict(kinetic=e,interaction=u,temperature=t,wick=wick,cutoffs=values))
        assert abs(values['120']-wick)<1e-10 and abs(values['120']-values['80'])<1e-10
    phonon=[]
    for t in [.04,.02,.01,.005]:
        actual=good.variance_density(t,1.,320);limit=2*np.pi**2*t**5/45
        phonon.append(dict(temperature=t,variance=actual,leading_phonon=limit,ratio=actual/limit))
    assert abs(phonon[-1]['ratio']-1)<.002
    assert all(abs(phonon[i+1]['ratio']-1)<abs(phonon[i]['ratio']-1) for i in range(len(phonon)-1))
    noiseless=[]
    for gain in [.7,.85,1.08,1.2,1.3]:
        values=ref.predict(inputs,gain);model=good.Model().fit([dict(input=e,value=float(v),sigma=1e-5) for e,v in zip(inputs,values)])
        noiseless.append(dict(true=gain,fitted=model.variance_gain))
    information=float(np.sum((good.predict_at(inputs,1.)/sigma)**2))
    controls_depletion=[]
    for t in [.08,.9]:
        for u in [0.,1.]:controls_depletion.append(dict(temperature=t,interaction=u,fraction=good.depletion(t,u)/10))
    report=dict(task='bogoliubov-momentum',revision=1,noise_trials=count,controls=controls,noise_samples=samples,checks=dict(reference_absolute=reference_error,reference_relative=reference_relative,oracle192_to320=refinement,ground_depletion_error=ground_depletion,ideal_variance_error=ideal_variance,ideal_calibration_equivalence=ideal_equivalence,fock_pair_checks=fock,phonon_limit=phonon,noiseless_fits=noiseless,weighted_fit_curvature=2*information,depletion_fractions=controls_depletion))
    report['summary']=dict(calibration_passes=sum(s['chi2']<1.5 for s in samples),parameter_passes=sum(abs(s['variance_gain']/1.08-1)<.03 for s in samples),oracle_passes=sum(max(s['oracle'].values())<.04 for s in samples),shortcut_prediction_passes=sum(max(s['shortcut'].values())<.04 for s in samples),oracle_worst=max(max(s['oracle'].values()) for s in samples),shortcut_interacting_best=min(s['shortcut'][key] for s in samples for key in hidden if key!='ideal_and_ground'),shortcut_anchor_worst=max(s['shortcut']['ideal_and_ground'] for s in samples),max_chi2=max(s['chi2'] for s in samples))
    assert report['summary']['calibration_passes']==count and report['summary']['parameter_passes']==count
    assert report['summary']['oracle_passes']==count and report['summary']['shortcut_prediction_passes']==0
    assert report['summary']['shortcut_anchor_worst']<.04
    assert reference_relative<1e-6 and refinement<1e-10 and ground_depletion<1e-10
    assert ideal_variance<1e-10 and ideal_equivalence<1e-10
    assert information>0 and max(abs(x['fitted']-x['true']) for x in noiseless)<1e-9
    return report


def local_checks():
    results={}
    for label,source in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/bogoliubov_momentum_baseline.py')]:
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);shutil.copy2(source,p/'model.py');shutil.copy2(TASK/'environment/test_public.py',p/'test_public.py');shutil.copytree(TASK/'environment/data',p/'data');shutil.copytree(TASK/'tests',p/'private')
            result=subprocess.run([sys.executable,'-m','pytest','-q','test_public.py','private/test_hidden.py'],cwd=p,env=dict(os.environ,PYTHONPATH=str(p),OPENBLAS_NUM_THREADS='1'),text=True,capture_output=True)
            results[label]=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            assert (result.returncode==0)==(label=='oracle')
            if label=='shortcut':assert '3 failed, 5 passed' in result.stdout
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--count',type=int,default=256);args=parser.parse_args()
    if args.generate:ref.generate_data()
    report=validate(args.count);report['local_controls']=local_checks()
    (ROOT/'results/bogoliubov-momentum-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(summary=report['summary'],checks=report['checks'],local_controls=report['local_controls']),indent=2))
