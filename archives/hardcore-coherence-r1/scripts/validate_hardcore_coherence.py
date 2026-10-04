"""Science, fixed-noise calibration and local harness checks."""
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
from scipy.linalg import expm_frechet
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/hardcore-coherence'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def error(value,truth):return float(np.linalg.norm(value-truth)/np.linalg.norm(truth))


def science(good,bad,ref):
    rng=np.random.default_rng(119199);cases=[]
    for j in [.8,1.,1.2]:
        for trap in [0.,.15]:
            for tilt in [-.3,.3]:
                for duration in [0.,.4,2.]:cases.append((j,trap,tilt,duration))
    cases += [(rng.uniform(.8,1.2),rng.uniform(0,.15),rng.uniform(-.3,.3),rng.uniform(0,2)) for _ in range(36)]
    rows=[]
    for j,a,b,t in cases:
        rho=good.correlation_matrix(j,a,b,t);short=bad.correlation_matrix(j,a,b,t);truth,psi,H=ref.exact_fock(j,a,b,t)
        reverse=good.correlation_matrix(j,a,b,-t)
        qs=np.linspace(-np.pi,np.pi,33,endpoint=False)
        signals=np.array([ref.momentum(rho,q) for q in qs]);other=np.array([ref.momentum(short,q) for q in qs])
        density_error=float(abs(rho.diagonal()-short.diagonal()).max())
        row=dict(hopping=j,trap=a,tilt=b,duration=t,reference_error=float(abs(rho-truth).max()),density_equivalence=density_error,
                 nearest_bond_equivalence=float(abs(np.diag(rho,1)-np.diag(short,1)).max()),hermiticity_error=float(abs(rho-rho.conj().T).max()),
                 trace_error=float(abs(np.trace(rho)-3)),minimum_eigenvalue=float(np.linalg.eigvalsh(rho).min()),minimum_momentum=float(signals.min()),
                 shortcut_minimum_momentum=float(other.min()),momentum_integral_error=float(abs(signals.mean()*8-3)),
                 time_reversal_density_error=float(abs(rho.diagonal()-reverse.diagonal()).max()),
                 time_reversal_momentum_error=max(abs(ref.momentum(rho,q)-ref.momentum(reverse,-q)) for q in qs),
                 energy_error=float(abs(np.vdot(psi,H@psi).real-np.sum(np.diag(good.hamiltonian(j,a,b))[list(good.OCCUPIED)]))))
        assert row['reference_error']<3e-12 and density_error<1e-12 and row['nearest_bond_equivalence']<1e-12
        assert row['hermiticity_error']<1e-12 and row['trace_error']<1e-12 and row['minimum_eigenvalue']>-1e-12
        assert min(row['minimum_momentum'],row['shortcut_minimum_momentum'])>-1e-12 and row['momentum_integral_error']<1e-12
        assert row['time_reversal_density_error']<1e-12 and row['time_reversal_momentum_error']<1e-12 and row['energy_error']<1e-12
        rows.append(row)
    # Direct many-body time integration also checks the reference eigensolver.
    ode=[]
    for j,a,b,t in [(.8,0.,-.3,2.),(1.2,.15,.3,2.),(1.03,.08,.13,.77)]:
        _,psi,H=ref.exact_fock(j,a,b,t);_,initial,_=ref.exact_fock(j,a,b,0.)
        out=solve_ivp(lambda time,y:-1j*H@y,[0,t],initial,method='DOP853',rtol=2e-12,atol=2e-13).y[:,-1]
        ode.append(float(max(abs(out-psi))))
    assert max(ode)<2e-11
    # Derivatives are independently obtained from a matrix-exponential Frechet derivative.
    # Sampling is numerical evidence over the declared interval, not an interval proof.
    cal=ref.calibration_inputs()[:24];grid=np.linspace(.8,1.2,121);slopes=[]
    direction=np.diag(np.full(7,-1.),1)+np.diag(np.full(7,-1.),-1)
    for j in grid:
        for e in cal:
            H=good.hamiltonian(j,e['trap'],e['tilt']);t=e['duration'];U,dU=expm_frechet(-1j*t*H,-1j*t*direction)
            site=e['site'];P=U[site,list(good.OCCUPIED)];dP=dU[site,list(good.OCCUPIED)]
            slopes.append(float(2*np.vdot(P,dP).real))
    assert min(slopes)>.02
    recovery=[];profiles=[]
    predictions=np.array([bad.predict_at(cal,j) for j in grid])
    for true in np.linspace(.8,1.2,41):
        truth=bad.predict_at(cal,true)
        fit=minimize_scalar(lambda j:np.sum((bad.predict_at(cal,j)-truth)**2),bounds=(.8,1.2),method='bounded',options={'xatol':1e-12}).x
        recovery.append(dict(true=float(true),fit=float(fit),error=float(abs(true-fit))))
        if true in [.8,1.,1.2]:
            values=np.sum((predictions-truth)**2,axis=1);index=int(np.argmin(values))
            assert np.all(np.diff(values[:index+1])<0) and np.all(np.diff(values[index:])>0)
            profiles.append(dict(true=float(true),hopping_grid=grid.tolist(),objective=values.tolist(),minimum_index=index))
    assert max(r['error'] for r in recovery)<1e-7
    groups=ref.hidden_inputs();full_range=[]
    for j in np.linspace(.8,1.2,41):
        gaps={k:error(bad.predict_at(e,j),good.predict_at(e,j)) for k,e in groups.items() if k.endswith('release')}
        assert min(gaps.values())>.15
        full_range.append(dict(hopping=float(j),errors=gaps))
    cal_equivalence=max(float(abs(good.predict_at(cal,j)-bad.predict_at(cal,j)).max()) for j in grid)
    assert cal_equivalence<1e-12
    return dict(domain_checks=rows,reference_error_max=max(r['reference_error'] for r in rows),many_body_ode_error_max=max(ode),
                minimum_sampled_calibration_derivative=min(slopes),derivative_grid_points=len(grid),identifiability_scope='numerical full-interval sampling and objective profiles, not a rigorous interval proof',
                calibration_equivalence=cal_equivalence,parameter_recovery=recovery,objective_profiles=profiles,full_range_hidden=full_range,
                full_range_shortcut_error_min=min(min(r['errors'].values()) for r in full_range))


def local_controls():
    result={}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/hardcore_coherence_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='hardcore-local-') as directory:
            work=Path(directory);shutil.copytree(TASK/'environment',work/'app');shutil.copytree(TASK/'tests',work/'tests');shutil.copy2(path,work/'app/model.py')
            env=dict(os.environ,PYTHONPATH=str(work/'app'),MODEL_PATH=str(work/'app/model.py'),METRICS_PATH=str(work/'metrics.json'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.perf_counter();run=subprocess.run([sys.executable,'-m','pytest','-q',str(work/'app/test_public.py'),str(work/'tests/test_hidden.py')],cwd=work,env=env,capture_output=True,text=True)
            result[label]=dict(returncode=run.returncode,seconds=time.perf_counter()-start,stdout=run.stdout,stderr=run.stderr,metrics=json.loads((work/'metrics.json').read_text()))
            assert run.returncode==(0 if label=='oracle' else 1),run.stdout
            assert ('10 passed' in run.stdout if label=='oracle' else '3 failed, 7 passed' in run.stdout),run.stdout
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();start=time.perf_counter()
    good=load(TASK/'solution/model.py','oracle');bad=load(ROOT/'scripts/hardcore_coherence_baseline.py','shortcut');ref=load(TASK/'tests/reference.py','reference')
    meta=json.loads((TASK/'tests/metadata.json').read_text());inputs=ref.calibration_inputs();truth=ref.predict(inputs);sigma=meta['sigma']
    if args.generate:
        values=truth+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,values)]
        for folder in ['environment','tests']:(TASK/folder/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();htruth={k:ref.predict(e) for k,e in hidden.items()};diagnostics=[k for k in hidden if k.endswith('release')]
    def score(model):return {k:error(model.predict(e),htruth[k]) for k,e in hidden.items()}
    def chi(model,rows):return float(np.sum(((model.predict(inputs)-np.array([r['value'] for r in rows]))/sigma)**2)/(len(rows)-1))
    report=dict(task='hardcore-coherence',revision=1,status='staged_unevaluated',metadata=meta,controls={})
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records);scores=score(model);report['controls'][label]=dict(hopping=model.hopping,calibration_chi2=chi(model,records),hidden=scores)
        assert chi(model,records)<1.5 and abs(model.hopping-1)<.03
        assert scores['density_anchors']<.04 and scores['initial_momentum']<.04
        assert max(scores.values())<.04 if label=='oracle' else min(scores[k] for k in diagnostics)>.04
    rng=np.random.default_rng(meta['noise_validation_seed']);parameters=[];chis=[];oracle_errors=[];shortcut_errors=[];group_short={k:[] for k in diagnostics}
    for draw in range(256):
        rows=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,truth+rng.normal(0,sigma,len(inputs)))];a=good.Model().fit(rows);b=bad.Model().fit(rows)
        assert abs(a.hopping-b.hopping)<1e-7
        sa,sb=score(a),score(b);parameters.append(a.hopping);chis.append(chi(a,rows));oracle_errors.append(max(sa.values()));shortcut_errors.append(min(sb[k] for k in diagnostics))
        for k in diagnostics:group_short[k].append(sb[k])
        assert chis[-1]<1.5 and abs(a.hopping-1)<.03 and oracle_errors[-1]<.04 and shortcut_errors[-1]>.04
        assert sb['density_anchors']<.04 and sb['initial_momentum']<.04
    report['noise']=dict(realizations=256,oracle_passes=256,shortcut_rejections=256,calibration_passes=256,maximum_chi2=max(chis),maximum_parameter_relative_error=float(max(abs(np.array(parameters)-1))),oracle_error_max=max(oracle_errors),shortcut_error_min=min(shortcut_errors),shortcut_group_minimum={k:min(v) for k,v in group_short.items()})
    report['science']=science(good,bad,ref);report['local_controls']=local_controls();report['seconds']=time.perf_counter()-start
    files=[f for f in TASK.rglob('*') if f.is_file() and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts]+[Path(__file__),ROOT/'scripts/hardcore_coherence_baseline.py']
    report['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
    (ROOT/'results/hardcore-coherence-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'results/hardcore-coherence-r1-local-controls.json').write_text(json.dumps(report['local_controls'],indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['science','local_controls','source_sha256']},indent=2))


if __name__=='__main__':main()
