from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
S=Path('/Users/yp9846/Desktop/PhysLiteBench/staging/overnight-20261011');T=S/'tasks/superconducting-heat';A=S/'predecessors/superconducting-heat-r10'
def load(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(a,b):return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),1e-100))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args();tick=time.monotonic()
    good=load('good',T/'solution/model.py');bad=load('bad',S/'scripts/superconducting_heat_baseline.py');ref=load('ref',T/'tests/reference.py');old=load('old',A/'task/solution/model.py')
    prior=json.loads((A/'validation.json').read_text());assert all(sha(A/'task'/p)==h for p,h in prior['source_sha256'].items())
    assert (T/'solution/model.py').read_text().startswith((A/'task/solution/model.py').read_text())
    assert (T/'tests/test_hidden.py').read_bytes()==(A/'task/tests/test_hidden.py').read_bytes()
    records=json.loads((T/'environment/data/calibration.json').read_text());assert records==json.loads((T/'tests/data/calibration.json').read_text()) and sha(T/'environment/data/calibration.json')==sha(A/'task/environment/data/calibration.json')
    inputs=[r['input'] for r in records];y=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);hidden=ref.hidden_inputs();truth={}
    for name,es in hidden.items():
        truth[name]=ref.predict(es,45.);print('Independent group',name,round(time.monotonic()-tick,1),flush=True)
    frozen_path=T/'tests/data/hidden_truth.json'
    if args.freeze:frozen_path.write_text(json.dumps(dict(reference_sha256=sha(T/'tests/reference.py'),description='Independent Fock/Bogoliubov channels and nonlinear finite-counting-field stationary jump-Hamiltonian saddle; central third difference at4nonzero counting fields. Prior heat and noise references retained.',true_parameter=45.,groups={n:dict(inputs=hidden[n],truth=v.tolist()) for n,v in truth.items()}),indent=2)+'\n')
    frozen=json.loads(frozen_path.read_text());assert frozen['reference_sha256']==sha(T/'tests/reference.py') and all(rel(truth[n],np.array(frozen['groups'][n]['truth']))<1e-12 for n in hidden)
    thirds=[n for n in hidden if n.endswith('_third')]
    def scores(m,g):return {n:rel(m.predict_at(es,g),truth[n]) for n,es in hidden.items()}
    controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.Model().fit(records);res=(fit.predict(inputs)-y)/sigma;start=time.monotonic();row=dict(parameter=fit.conductance,calibration_chi2=float(res@res)/(len(records)-1),hidden=scores(m,fit.conductance),seconds=time.monotonic()-start)
        assert abs(fit.conductance/45.-1)<.03 and row['calibration_chi2']<1.5 and row['seconds']<45
        if name=='oracle':assert max(row['hidden'].values())<.04,row
        else:assert min(row['hidden'][n] for n in thirds)>.04,row
        controls[name]=row
    print('CONTROLS',json.dumps(controls),flush=True)
    rng=np.random.default_rng(2026101121);domain=[]
    for i in range(24):
        left,right=rng.uniform(.25,1.05,2);phase=float(rng.uniform(-np.pi,np.pi));amplitude=float(rng.uniform(.3,1.));drive=float(rng.uniform(*((.1,.5) if i%2 else (7.,12.))));rho=float(rng.uniform(0,1));temperature=float(rng.uniform(12,24));secondary=float(rng.uniform(0,1));quadratic=float(rng.uniform(-.2,.2));momentum=float(rng.uniform(-.6,.6));conductance=float(rng.uniform(20,80))
        if i<8:left=(.25,1.05)[i&1];right=(.25,1.05)[(i>>1)&1];amplitude=(.3,1.)[(i>>2)&1];drive=(.1,.5,7.,12.)[i%4];rho=(0.,1.)[i&1];temperature=(12.,24.)[(i>>1)&1];secondary=(0.,1.)[(i>>2)&1];quadratic=(-.2,.2)[i&1];momentum=(-.6,.6)[(i>>1)&1];conductance=(20.,80.)[i&1]
        e=ref.gate_experiment(left=left,right=right,phase=phase,amplitude=amplitude,drive=drive,rho=rho,temperature=temperature,secondary=secondary,quadratic=quadratic,momentum=momentum);e['statistic']='gate_third'
        a=good.gate_third(e,conductance);b=ref.saddle_reference(e,conductance);fine=good.gate_third(e,conductance,nodes=288,count=80)
        row=dict(experiment=e,conductance=conductance,oracle=a,independent=b,independent_relative_error=rel(a['third'],b['third']),quadrature_basis_relative_error=rel(a['third'],fine['third']))
        assert row['independent_relative_error']<.002 and row['quadrature_basis_relative_error']<2e-4,row
        if i in [0,7,15,23]:
            refined=ref.saddle_reference(e,conductance,nodes=384,count=80,step=.00025);row['reference_refinement_relative_error']=rel(a['third'],refined['third']);assert row['reference_refinement_relative_error']<.002,row
        # Higher statistics extend the exact preceding model; no previous
        # observable may change as a side effect of this revision.
        for statistic in ['mean','gate_noise']:
            anchor=dict(e,statistic=statistic);before=old.predict_at([anchor],conductance);after=good.predict_at([anchor],conductance);assert np.array_equal(before,after)
        domain.append(row)
        good.prepare.cache_clear();bad.prepare.cache_clear();old.prepare.cache_clear();ref.fock_channels.cache_clear()
        if (i+1)%4==0:print('Third domain',i+1,'of24',round(time.monotonic()-tick,1),flush=True)
    limits=[]
    e=ref.gate_experiment(amplitude=0.);a=good.gate_third(e,45.);b=ref.saddle_reference(e,45.);assert a['third']==b['third']==bad.gate_third(e,45.)['third']==0.;limits.append(dict(kind='static_barrier',third=0.))
    e=ref.gate_experiment(amplitude=1.,drive=9.);old_tau=good.NORMAL_POLARIZATION_TIME
    try:
        good.NORMAL_POLARIZATION_TIME=old_tau/1e6;good.stationary.cache_clear();a=good.gate_third(e,45.);difference=abs(a['third']/a['screened_poisson']-1);assert difference<1e-5,difference;limits.append(dict(kind='rapid_spin_relaxation',relative_nonlinear_correction=difference,third=a['third']))
    finally:good.NORMAL_POLARIZATION_TIME=old_tau;good.stationary.cache_clear()
    # For the zero spin-loss conservation law, w=drive/2*sum(ds) on each
    # tunnel event means its integrated work is a bounded spin difference.
    # The corresponding counting conjugate cancels every mark identically.
    e=ref.gate_experiment(drive=7.);args0=good.oscillator_arguments(e);kw=dict(secondary_coupling=e['secondary_coupling'],quadratic_coupling=e['quadratic_coupling'],momentum_coupling=e['momentum_coupling']);state=good.stationary(*args0,45.,**kw);ch=good.prepare(*args0,window=state['search_bound'],**kw)[2];mark=ch[6]-args0[4]/2*ch[-1].sum(axis=1);assert max(abs(mark))<1e-13;limits.append(dict(kind='no_spin_loss_boundary_work_identity',maximum_event_mark=float(max(abs(mark)))))
    clean=good.predict_at(inputs,45.);rng=np.random.default_rng(43023);fits=[]
    for _ in range(256):
        noisy=clean+rng.normal(size=len(records))*sigma;sample=[dict(r,value=float(v)) for r,v in zip(records,noisy)];a=good.Model().fit(sample);b=bad.Model().fit(sample);res=(a.predict(inputs)-noisy)/sigma;chi=float(res@res)/(len(records)-1)
        assert abs(a.conductance/45.-1)<.03 and abs(a.conductance-b.conductance)<1e-10 and chi<1.5;fits.append(dict(parameter=a.conductance,chi2=chi))
    params=np.array([r['parameter'] for r in fits]);sensitivity=[]
    for g in np.quantile(params,np.linspace(0,1,9)):
        a=scores(good,g);b=scores(bad,g);assert max(a.values())<.04 and min(b[n] for n in thirds)>.04;sensitivity.append(dict(parameter=float(g),oracle=a,shortcut=b));print('Third fits',len(sensitivity),'of9',flush=True)
    report=dict(revision=11,calibration_regenerated=False,inherited_r10_validation_sha256=sha(A/'validation.json'),controls=controls,physics=dict(domain=domain,limits=limits),noise=dict(seed=43023,trials=256,method='256 actual calibration fits using the same unchanged-calibration noise seed as R10; all hidden predictions at nine fitted-parameter quantiles including extrema.',fit_min=float(params.min()),fit_max=float(params.max()),max_chi2=max(r['chi2'] for r in fits),hidden_sensitivity=sensitivity),seconds=time.monotonic()-tick,source_sha256={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and not any(k in p.parts for k in ['__pycache__','.pytest_cache'])})
    (S/'results/superconducting-heat-r11-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',round(report['seconds'],1),flush=True)
if __name__=='__main__':main()
