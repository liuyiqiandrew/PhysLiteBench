"""Validate the staged rotor's physical limit, calibration and isolated controls."""
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

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/geometric-rotor'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def full_fourier_kinetic(inertia,e,gap=64.,cutoff=12):
    """Full laboratory-basis matrix, without splitting it into invariant blocks."""
    n=np.arange(-cutoff,cutoff+1);size=len(n)
    kinetic=np.tile(n*n/(2*inertia),2)
    h=np.diag(kinetic).astype(complex)
    h[:size,:size]+=gap*np.cos(e['tilt'])*np.eye(size)
    h[size:,size:]-=gap*np.cos(e['tilt'])*np.eye(size)
    for a,up in enumerate(n):
        down=up+e['winding']
        where=np.flatnonzero(n==down)
        if len(where):
            b=size+where[0]
            h[a,b]=h[b,a]=gap*np.sin(e['tilt'])
    energies,vectors=np.linalg.eigh(h)
    population=np.exp(-(energies-energies.min())/e['temperature']);population/=population.sum()
    expectation=np.sum(abs(vectors)**2*kinetic[:,None],axis=0)
    return float(population@expectation)


def science(good,bad,ref):
    cases=list(itertools.product([.8,1.2],[.04,.4],[0,1,2,3,4],[0.,np.pi/3,np.pi/2,2*np.pi/3,np.pi]))
    rng=np.random.default_rng(173092)
    cases += [(rng.uniform(.8,1.2),rng.uniform(.04,.4),int(rng.integers(0,5)),rng.uniform(0,np.pi)) for _ in range(64)]
    ref_errors=[];refinements=[];cutoff_errors=[]
    for inertia,t,q,angle in cases:
        e=ref.experiment(t,q,angle);a=good.predict_at([e],inertia)[0]
        b=ref.kinetic_energy(inertia,t,q,angle)
        c=ref.kinetic_energy(inertia,t,q,angle,2048.)
        ref_errors.append(abs(a-b));refinements.append(abs(b-c))
        cutoff_errors.append(abs(ref.finite_gap(inertia,t,q,angle,64.,24)-ref.finite_gap(inertia,t,q,angle,64.,36)))
    assert max(ref_errors)<1e-7 and max(refinements)<1e-7 and max(cutoff_errors)<1e-12
    matrix_errors=[]
    for inertia,t,q,angle in cases[-8:]:
        e=ref.experiment(t,q,angle)
        matrix_errors.append(abs(full_fourier_kinetic(inertia,e)-ref.finite_gap(inertia,t,q,angle,64.)))
    assert max(matrix_errors)<1e-10
    inputs=ref.calibration_inputs()[0:8]
    equivalence=max(float(abs(good.predict_at(inputs,i)-bad.predict_at(inputs,i)).max()) for i in [.8,1.,1.2])
    recovery=[];minima=[];minimum_descent=np.inf
    for i in [.8,1.,1.2]:
        exact=good.predict_at(inputs,i)
        rows=[dict(input=e,value=float(y),sigma=.0005) for e,y in zip(inputs,exact)]
        recovery.append(abs(bad.Model().fit(rows).inertia-i))
        grid=np.linspace(.8,1.2,801)
        values=np.array([bad.predict_at(inputs,x) for x in grid])
        objective=np.sum((values-exact)**2,axis=1)
        minimum_descent=min(minimum_descent,float((-np.diff(values,axis=0)).min()))
        minima.append(int(np.sum((objective[1:-1]<objective[:-2])&(objective[1:-1]<objective[2:])))+int(objective[0]<objective[1])+int(objective[-1]<objective[-2]))
    assert equivalence<1e-13 and max(recovery)<1e-7 and minima==[1,1,1] and minimum_descent>0
    zero_texture=[];gauge=[];high_temperature=[];low_temperature=[]
    for i in [.8,1.2]:
        for q in [0,1,2,3,4]:
            for theta in [0.,np.pi]:
                e=ref.experiment(.13,q,theta);zero_texture.append(abs(good.predict_at([e],i)[0]-good.predict_at([ref.experiment(.13,0,0.)],i)[0]))
        e=ref.experiment(.04,1,np.pi/2);low_temperature.append(abs(good.predict_at([e],i)[0]-.25/i))
        # High-T continuum value includes the exact diagonal kinetic scalar.
        e=ref.experiment(5.,1,1.2);target=2.5+np.sin(1.2)**2/(8*i)
        high_temperature.append(abs(good.predict_at([e],i)[0]-target))
    n=np.arange(-32,33)
    for t in [.04,.2]:
        for q in [1,2,4]:
            angle=1.13;inertia=1.07;a=.5*q*(1-np.cos(angle));scalar=q*q*np.sin(angle)**2/(8*inertia)
            en=(n-a-3)**2/(2*inertia)+scalar;w=np.exp(-(en-en.min())/t);w/=w.sum()
            gauge.append(abs(w@en-good.predict_at([ref.experiment(t,q,angle)],inertia)[0]))
    assert max(zero_texture+gauge)<1e-12 and max(high_temperature)<1e-10 and max(low_temperature)<1e-8
    return dict(domain_cases=len(cases),inverse_gap_limit_error=max(ref_errors),inverse_gap_refinement=max(refinements),
                angular_cutoff_error=max(cutoff_errors),full_unsplit_fourier_matrix_error=max(matrix_errors),
                calibration_equivalence=equivalence,noiseless_parameter_recovery=max(recovery),calibration_profile_minima=minima,
                minimum_calibration_energy_descent_per_grid_step=minimum_descent,zero_texture_error=max(zero_texture),
                integer_gauge_invariance_error=max(gauge),low_temperature_half_flux_error=max(low_temperature),high_temperature_limit_error=max(high_temperature))


def local_controls():
    out={}
    for label,source in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/geometric_rotor_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='geometric-rotor-control-') as tmp:
            p=Path(tmp);shutil.copytree(TASK/'environment',p/'app');shutil.copytree(TASK/'tests',p/'tests');shutil.copy2(source,p/'app/model.py')
            env=dict(os.environ,PYTHONPATH=str(p/'app'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.perf_counter();r=subprocess.run([sys.executable,'-m','pytest','-q',str(p/'app/test_public.py'),str(p/'tests/test_hidden.py')],cwd=p,env=env,capture_output=True,text=True)
            out[label]=dict(returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr)
            assert r.returncode==(0 if label=='oracle' else 1)
            assert ('8 passed' in r.stdout if label=='oracle' else '3 failed, 5 passed' in r.stdout)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    good=load(TASK/'solution/model.py','oracle');bad=load(ROOT/'scripts/geometric_rotor_baseline.py','shortcut');ref=load(TASK/'tests/reference.py','reference')
    meta=json.loads((TASK/'tests/metadata.json').read_text());sigma=meta['measurement_sigma'];true=ref.TRUE_PARAMETER;start=time.time()
    inputs=ref.calibration_inputs();exact=ref.predict(inputs)
    if args.generate:
        values=exact+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        for name in ['environment','tests']:(TASK/name/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();truth={k:ref.predict(es) for k,es in hidden.items()};diagnostics=list(hidden)[:-1]
    def scores(model):return {k:float(np.sqrt(np.mean((model.predict(es)-truth[k])**2)/np.mean(truth[k]**2))) for k,es in hidden.items()}
    def chi(model,rows):return float(np.sum(((model.predict(inputs)-np.array([r['value'] for r in rows]))/sigma)**2)/(len(rows)-1))
    report=dict(task='geometric-rotor',revision=1,status='staged_unevaluated',calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],controls={})
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records);s=scores(model)
        report['controls'][label]=dict(inertia=model.inertia,parameter_relative_error=abs(model.inertia/true-1),calibration_chi2=chi(model,records),hidden=s)
        assert chi(model,records)<1.5 and abs(model.inertia/true-1)<.03 and s['even_winding_anchors']<.04
        assert (max(s.values())<.04 if label=='oracle' else min(s[k] for k in diagnostics)>.04)
    rng=np.random.default_rng(meta['noise_seed']);chis=[];parameters=[];oracle_errors=[];shortcut_errors=[]
    for _ in range(256):
        rows=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a,b=good.Model().fit(rows),bad.Model().fit(rows)
        assert abs(a.inertia-b.inertia)<1e-10
        parameters.append(a.inertia);chis.append(chi(a,rows));oracle_errors.append(max(scores(a).values()));shortcut_errors.append(min(scores(b)[k] for k in diagnostics))
        assert chis[-1]<1.5 and abs(a.inertia/true-1)<.03 and oracle_errors[-1]<.04 and shortcut_errors[-1]>.04
    report['noise']=dict(realizations=256,calibration_passes=256,parameter_passes=256,oracle_passes=256,shortcut_rejections=256,maximum_chi2=max(chis),
                         maximum_relative_parameter_error=float(abs(np.array(parameters)/true-1).max()),parameter_range=[min(parameters),max(parameters)],oracle_hidden_max=max(oracle_errors),shortcut_hidden_min=min(shortcut_errors))
    report['science']=science(good,bad,ref);report['local_controls']=local_controls();report['seconds']=time.time()-start
    source=[TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/geometric_rotor_baseline.py',Path(__file__)]
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    (ROOT/'results/geometric-rotor-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
