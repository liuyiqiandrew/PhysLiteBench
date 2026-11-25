"""Validate the closed-ring revision without any agent evaluations."""
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

ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/ionic-current-loops'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def key(e):return json.dumps({k:e[k] for k in ['means','amplitudes','waves','phases']},sort_keys=True)
def rms(actual,truth):return float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))


def science(good,bad,ref):
    hidden=ref.hidden_inputs();es=sum(hidden.values(),[])
    agreement=[];refinement=[];oracle_refinement=[];mass=[];charge=[];emf=[];current_variation=[];entropy=[];concentration=[];short_emf=[]
    for e in es:
        agreement.append(abs(good.predict_at([e],1.13)[0]-ref.predict([e])[0]))
        refinement.append(abs(ref.predict([e],points=512)[0]-ref.predict([e],points=1024)[0]))
        x,c,E,flux,rate=good.fields(key(e));xf,cf,Ef,jf,rf=ref.fields(key(e),1024)
        fine=good.fields(key(e),513);coarse_signal=2*np.mean(rate[e['species']]*np.cos(e['detector']*x+e['detector_phase']));fine_signal=2*np.mean(fine[4][e['species']]*np.cos(e['detector']*fine[0]+e['detector_phase']));oracle_refinement.append(abs(coarse_signal-fine_signal))
        z=np.array([1.,1.,-1.]);d=np.array([1.,4.,1.]);j=z@flux
        mass.append(max(abs(np.mean(rate,axis=1))));charge.append(max(abs(z@rate)));emf.append(abs(np.mean(E)));current_variation.append(float(np.ptp(j)));concentration.append(float(c.min()))
        free_energy_rate=float(np.mean(np.sum(np.log(c)*rate,axis=0)));dissipation=float(np.mean(np.sum(flux*flux/(d[:,None]*c),axis=0)))
        entropy.append(abs(free_energy_rate+dissipation))
        short_emf.append(abs(np.mean(bad.fields(key(e))[2])))
        assert np.max(abs(z@rf))<1e-7 and abs(np.mean(Ef))<1e-12
    assert max(agreement)<2e-7 and max(refinement)<2e-7 and max(oracle_refinement)<1e-10
    assert max(mass)<1e-12 and max(charge)<1e-10 and max(emf)<1e-13 and max(current_variation)<1e-12
    assert max(entropy)<1e-10 and min(concentration)>.1
    # Independent periodic potential and analytic loop condition agree in current.
    currents=[];zero_currents=[];field_reversal=[]
    for e in hidden['phase_quadrature']:
        x,c,E,j,rate=good.fields(key(e));xf,cf,Ef,jf,rf=ref.fields(key(e),2048)
        currents.append(abs(np.mean(np.array([1.,1.,-1.])@j)-np.mean(np.array([1.,1.,-1.])@jf)))
        reversed_e=dict(e,phases=(-np.array(e['phases'])).tolist());jr=good.fields(key(reversed_e))[3]
        field_reversal.append(abs(np.mean(np.array([1.,1.,-1.])@(j+jr))))
    for e in ref.calibration_inputs():zero_currents.append(abs(np.mean(np.array([1.,1.,-1.])@good.fields(key(e))[3])))
    assert max(currents)<2e-5 and max(zero_currents)<1e-12 and max(field_reversal)<1e-12
    # Closed ring counterexample: sigma varies, E is curl-free locally, but
    # the zero-current closure has a nonzero period integral.
    baseline_cal=max(abs(good.predict_at(ref.calibration_inputs(),1)-bad.predict_at(ref.calibration_inputs(),1)))
    assert baseline_cal<1e-12 and max(short_emf)>.01
    # Check allowed corners without fitting or selecting new hidden cases.
    corners=[]
    for mean0,mean1 in [( .6,.6),(.6,1.4),(1.4,.6),(1.4,1.4)]:
        for phase in [0,.7,1.4,2.1,np.pi]:
            e=ref.experiment(means=(mean0,mean1),amplitudes=(.85*mean0,-.85*mean1),waves=(2,2),phases=(0,phase))
            a=good.predict_at([e],1)[0];b=ref.predict([e],1,points=1024)[0];corners.append(abs(a-b))
    assert max(corners)<5e-7
    return dict(oracle_reference_max=max(agreement),reference_refinement_max=max(refinement),oracle_refinement_max=max(oracle_refinement),integrated_mass_rate_max=max(mass),local_charge_rate_max=max(charge),mean_electric_field_max=max(emf),charge_current_variation_max=max(current_variation),free_energy_dissipation_identity_error=max(entropy),minimum_hidden_concentration=min(concentration),maximum_shortcut_mean_field=max(short_emf),independent_periodic_solve_current_error=max(currents),reflection_current_error=max(field_reversal),calibration_current_max=max(zero_currents),exact_calibration_equivalence=baseline_cal,public_corner_reference_error=max(corners))


def controls():
    out={}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/ionic_current_loops_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='ionic-ring-local-') as directory:
            work=Path(directory);shutil.copytree(TASK/'environment',work/'app');shutil.copytree(TASK/'tests',work/'tests');shutil.copy2(path,work/'app/model.py')
            env=dict(os.environ,PYTHONPATH=str(work/'app'),MODEL_PATH=str(work/'app/model.py'),METRICS_PATH=str(work/'metrics.json'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.perf_counter();result=subprocess.run([sys.executable,'-m','pytest','-q',str(work/'app/test_public.py'),str(work/'tests/test_hidden.py')],cwd=work,env=env,capture_output=True,text=True)
            out[label]=dict(returncode=result.returncode,seconds=time.perf_counter()-start,stdout=result.stdout,stderr=result.stderr,metrics=json.loads((work/'metrics.json').read_text()))
            assert result.returncode==(0 if label=='oracle' else 1)
            assert ('9 passed' in result.stdout if label=='oracle' else '3 failed, 6 passed' in result.stdout)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();started=time.perf_counter()
    good=load(TASK/'solution/model.py','oracle');bad=load(ROOT/'scripts/ionic_current_loops_baseline.py','shortcut');ref=load(TASK/'tests/reference.py','reference')
    meta=json.loads((TASK/'tests/metadata.json').read_text());sigma=meta['sigma'];inputs=ref.calibration_inputs();truth=ref.predict(inputs)
    if args.generate:
        observed=truth+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs));records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,observed)]
        for name in ['environment','tests']:(TASK/name/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();htruth={k:ref.predict(es) for k,es in hidden.items()};diagnostics=[k for k in hidden if k!='aligned_profiles']
    def score(m):return {k:rms(m.predict(es),htruth[k]) for k,es in hidden.items()}
    def chi(m,rows):return float(np.sum(((m.predict(inputs)-np.array([r['value'] for r in rows]))/sigma)**2)/(len(rows)-1))
    report=dict(task='ionic-current-loops',revision=2,status='staged_unevaluated',metadata=meta,calibration_records=len(records),controls={})
    for label,module in [('oracle',good),('shortcut',bad)]:
        m=module.Model().fit(records);s=score(m);report['controls'][label]=dict(diffusivity=m.diffusivity,calibration_chi2=chi(m,records),hidden=s)
        assert chi(m,records)<1.5 and abs(m.diffusivity/1.13-1)<.03 and s['aligned_profiles']<.04
        assert max(s.values())<.04 if label=='oracle' else min(s[k] for k in diagnostics)>.04
    rng=np.random.default_rng(meta['noise_validation_seed']);parameters=[];chis=[];oracle_errors=[];shortcut_errors=[]
    for _ in range(256):
        rows=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,truth+rng.normal(0,sigma,len(inputs)))];a,b=good.Model().fit(rows),bad.Model().fit(rows)
        assert abs(a.diffusivity-b.diffusivity)<1e-12
        parameters.append(a.diffusivity);chis.append(chi(a,rows));oracle_errors.append(max(score(a).values()));shortcut_errors.append(min(score(b)[k] for k in diagnostics))
        assert chis[-1]<1.5 and abs(a.diffusivity/1.13-1)<.03 and oracle_errors[-1]<.04 and shortcut_errors[-1]>.04
        assert score(b)['aligned_profiles']<.04
    report['noise']=dict(realizations=256,calibration_passes=256,parameter_passes=256,oracle_passes=256,shortcut_rejections=256,maximum_chi2=max(chis),maximum_relative_parameter_error=float(max(abs(np.array(parameters)/1.13-1))),parameter_range=[min(parameters),max(parameters)],oracle_hidden_max=max(oracle_errors),shortcut_hidden_min=min(shortcut_errors))
    coefficient=good.predict_at(inputs,1);report['calibration_information']=float(np.sum(coefficient**2/sigma**2));assert report['calibration_information']>1e5
    report['science']=science(good,bad,ref);report['local_controls']=controls();report['seconds']=time.perf_counter()-started
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]+[Path(__file__),ROOT/'scripts/ionic_current_loops_baseline.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (ROOT/'results/ionic-current-loops-r2-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
