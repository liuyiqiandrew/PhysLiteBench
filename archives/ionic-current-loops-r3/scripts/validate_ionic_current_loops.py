"""Scientific and local checks for the fully circulating ionic family revision."""
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
TASK=ROOT/'tasks/ionic-current-loops'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def error(actual,truth):
    return float(np.linalg.norm(actual-truth)/np.linalg.norm(truth))


def invariants(module,e):
    x,y,c,E,flux,rate=module.fields(tuple(e['amplitudes']),e['phase'])
    d=np.array([1.,4.,1.]);z=np.array([1.,1.,-1.])
    current=np.einsum('i,ijab->jab',z,flux)
    free_energy_rate=float(np.mean(np.sum(np.log(c)*rate,axis=0)))
    dissipation=float(np.mean(np.sum(flux**2/(d[:,None,None,None]*c[:,None]),axis=(0,1))))
    return dict(mass_rate=float(abs(rate.mean(axis=(1,2))).max()),charge_rate=float(abs(np.einsum('i,iab->ab',z,rate)).max()),
                mean_field=E.mean(axis=(1,2)).tolist(),mean_current=current.mean(axis=(1,2)).tolist(),
                current_rms=float(np.sqrt(np.mean(current**2))),minimum_concentration=float(c.min()),
                free_energy_rate=free_energy_rate,dissipation=dissipation,
                entropy_balance_error=abs(free_energy_rate+dissipation))


def science(good,bad,ref):
    hidden=ref.hidden_inputs();inputs=sum(hidden.values(),[])
    oracle=good.predict_at(inputs,1.13);truth=ref.predict(inputs)
    refined=ref.predict(inputs,points=96)
    spectral=[];checks=[]
    for e in inputs:
        x,y,_,_,_,rates=good.fields(tuple(e['amplitudes']),e['phase'],95)
        m,n=e['detector'];value=2*np.mean(rates[e['species']]*np.cos(m*x+n*y+np.pi/4))*1.13
        spectral.append(abs(value-good.predict_at([e],1.13)[0]))
        checks.append(dict(input=e,oracle=invariants(good,e),shortcut=invariants(bad,e)))
    assert max(abs(oracle-truth))<3e-6
    assert max(abs(refined-truth))<3e-6
    assert max(spectral)<1e-9
    for row in checks:
        for label in ['oracle','shortcut']:
            r=row[label]
            assert r['mass_rate']<1e-12 and r['charge_rate']<2e-9
            assert r['entropy_balance_error']<2e-9 and r['free_energy_rate']<0
            assert r['minimum_concentration']>.049
        assert max(abs(np.array(row['oracle']['mean_field'])))<1e-13
        assert max(abs(np.array(row['shortcut']['mean_current'])))<1e-11
    cal=ref.calibration_inputs();equivalence=float(max(abs(good.predict_at(cal,1)-bad.predict_at(cal,1))))
    assert equivalence<1e-12
    # Earlier pointwise-zero-current closure is tested against the same full
    # calibration, allowing its own best common diffusivity.
    proto=load(ROOT/'prototypes/prototype.py','ionic_prototype')
    old=[]
    cached={}
    for e in cal:
        key=tuple(e['amplitudes'])
        if key not in cached:cached[key]=proto.spectral(key,0.)
        old.append(proto.rate(cached[key],e['species'],e['detector'],np.pi/4,'old_local'))
    old=np.array(old);correct=good.predict_at(cal,1.13);best=float(old@correct/(old@old))
    residual=best*old-correct
    old_expected_chi=float(np.mean((residual/.001)**2)+1)
    assert old_expected_chi>2.5
    cal_currents=[invariants(good,ref.experiment(key,0)) for key in cached]
    assert min(r['current_rms'] for r in cal_currents)>.02
    # The full source elliptic implementation matches an independent conservative
    # finite-volume implementation of its own mean-current boundary ensemble.
    source_reference=[]
    for e in hidden['phase_sweep'][::2]:
        values=[proto.finite_volume(tuple(e['amplitudes']),e['phase'],e['species'],e['detector'],np.pi/4,n=n,closed=False) for n in [64,128]]
        source_reference.append(abs((4*values[1]-values[0])/3-bad.predict_at([e],1)[0]))
    assert max(source_reference)<3e-6
    # Public-domain controls, including detector signs/species and extrema.
    rng=np.random.default_rng(119163);corners=[]
    settings=[(.5,.2,.6,.15),(.65,.3,.7,.25),(.5,.3,.7,.15),(.65,.2,.6,.25)]
    for a in settings:
        for phase in [0.,.9,1.8,np.pi]:
            e=ref.experiment(a,phase,int(rng.integers(0,3)),[(1,0),(0,1),(1,1),(-1,2)][len(corners)%4])
            delta=abs(good.predict_at([e],1)[0]-ref.predict([e],1,points=96)[0]);corners.append(delta)
    assert max(corners)<3e-6
    reflection=[]
    for e in hidden['phase_sweep'][::2]:
        plus=invariants(good,e)['mean_current'];minus=invariants(good,dict(e,phase=-e['phase']))['mean_current']
        reflection.append(max(abs(np.array(plus)+minus)))
    assert max(reflection)<1e-12
    coefficient=good.predict_at(cal,1);recovery=[]
    for D in np.linspace(.5,2,41):
        records=[dict(input=e,value=float(v),sigma=.001) for e,v in zip(cal,D*coefficient)]
        recovery.append(abs(good.Model().fit(records).diffusivity-D))
    assert max(recovery)<1e-12
    return dict(independent_hidden_error=float(max(abs(oracle-truth))),finite_volume_refinement=float(max(abs(refined-truth))),spectral_refinement=max(spectral),
                calibration_equivalence=equivalence,minimum_calibration_local_current=min(r['current_rms'] for r in cal_currents),
                earlier_zero_current_best_parameter=best,earlier_zero_current_calibration_rms=float(np.sqrt(np.mean(residual**2))),
                earlier_zero_current_expected_chi2=old_expected_chi,source_independent_ensemble_error=max(source_reference),
                corner_reference_error=max(corners),reflection_current_error=max(reflection),full_range_parameter_recovery=max(recovery),
                conservation_passivity=checks)


def local_controls():
    result={}
    for label,model in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/ionic_current_loops_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='ionic-r3-local-') as directory:
            work=Path(directory);shutil.copytree(TASK/'environment',work/'app');shutil.copytree(TASK/'tests',work/'tests');shutil.copy2(model,work/'app/model.py')
            env=dict(os.environ,PYTHONPATH=str(work/'app'),MODEL_PATH=str(work/'app/model.py'),METRICS_PATH=str(work/'metrics.json'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.perf_counter();run=subprocess.run([sys.executable,'-m','pytest','-q',str(work/'app/test_public.py'),str(work/'tests/test_hidden.py')],cwd=work,env=env,capture_output=True,text=True)
            result[label]=dict(returncode=run.returncode,seconds=time.perf_counter()-start,stdout=run.stdout,stderr=run.stderr,metrics=json.loads((work/'metrics.json').read_text()))
            assert run.returncode==(0 if label=='oracle' else 1),run.stdout
            assert ('9 passed' in run.stdout if label=='oracle' else '3 failed, 6 passed' in run.stdout),run.stdout
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();start=time.perf_counter()
    good=load(TASK/'solution/model.py','oracle');bad=load(ROOT/'scripts/ionic_current_loops_baseline.py','shortcut');ref=load(TASK/'tests/reference.py','reference')
    meta=json.loads((TASK/'tests/metadata.json').read_text());inputs=ref.calibration_inputs();truth=ref.predict(inputs);sigma=meta['sigma']
    if args.generate:
        observed=truth+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,observed)]
        for folder in ['environment','tests']:(TASK/folder/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();htruth={k:ref.predict(e) for k,e in hidden.items()};diagnostics=[k for k in hidden if k!='symmetric_anchors']
    def score(model):return {k:error(model.predict(es),htruth[k]) for k,es in hidden.items()}
    def chi(model,rows):return float(np.sum(((model.predict(inputs)-np.array([r['value'] for r in rows]))/sigma)**2)/(len(rows)-1))
    report=dict(task='ionic-current-loops',revision=3,status='staged_unevaluated',metadata=meta,controls={})
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records);scores=score(model);report['controls'][label]=dict(diffusivity=model.diffusivity,calibration_chi2=chi(model,records),hidden=scores)
        assert chi(model,records)<1.5 and abs(model.diffusivity/1.13-1)<.03 and scores['symmetric_anchors']<.04
        assert max(scores.values())<.04 if label=='oracle' else min(scores[k] for k in diagnostics)>.04
    rng=np.random.default_rng(meta['noise_validation_seed']);parameters=[];chis=[];oracle_errors=[];shortcut_errors=[]
    for _ in range(256):
        rows=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,truth+rng.normal(0,sigma,len(inputs)))];a=good.Model().fit(rows);b=bad.Model().fit(rows)
        assert abs(a.diffusivity-b.diffusivity)<1e-12
        parameters.append(a.diffusivity);chis.append(chi(a,rows));oracle_errors.append(max(score(a).values()));shortcut_errors.append(min(score(b)[k] for k in diagnostics))
        assert chis[-1]<1.5 and abs(a.diffusivity/1.13-1)<.03 and oracle_errors[-1]<.04 and shortcut_errors[-1]>.04
        assert score(b)['symmetric_anchors']<.04
    report['noise']=dict(realizations=256,oracle_passes=256,shortcut_rejections=256,calibration_passes=256,maximum_chi2=max(chis),maximum_parameter_relative_error=float(max(abs(np.array(parameters)/1.13-1))),oracle_error_max=max(oracle_errors),shortcut_error_min=min(shortcut_errors))
    report['science']=science(good,bad,ref);report['local_controls']=local_controls();report['seconds']=time.perf_counter()-start
    files=[f for f in TASK.rglob('*') if f.is_file() and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts]+[Path(__file__),ROOT/'scripts/ionic_current_loops_baseline.py']
    report['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
    (ROOT/'results/ionic-current-loops-r3-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['science','local_controls','source_sha256']},indent=2))


if __name__=='__main__':main()
