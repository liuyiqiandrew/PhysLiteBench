from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
S=Path('/Users/yp9846/Desktop/PhysLiteBench/staging/overnight-20261011');T=S/'tasks/superconducting-heat';A=S/'predecessors/superconducting-heat-r9'
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--freeze',action='store_true');args=pa.parse_args();tick=time.monotonic()
    good=load('good',T/'solution/model.py');bad=load('bad',S/'scripts/superconducting_heat_baseline.py');ref=load('ref',T/'tests/reference.py');prior=json.loads((A/'validation.json').read_text())
    assert all(sha(A/'task'/p)==h for p,h in prior['source_sha256'].items())
    assert (T/'solution/model.py').read_text().startswith((A/'task/solution/model.py').read_text())
    assert (T/'tests/reference.py').read_text().startswith((A/'task/tests/reference.py').read_text())
    assert 'spin_jacobian' not in (T/'environment/model.py').read_text()
    assert sha(T/'environment/data/calibration.json')==sha(A/'task/environment/data/calibration.json')==sha(T/'tests/data/calibration.json')
    records=json.loads((T/'environment/data/calibration.json').read_text());inputs=[r['input'] for r in records];y=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);hidden=ref.hidden_inputs();truth={}
    for name,es in hidden.items():truth[name]=ref.predict(es,45.);print('Independent group',name,round(time.monotonic()-tick,1),flush=True)
    frozen_path=T/'tests/data/hidden_truth.json'
    if args.freeze:frozen_path.write_text(json.dumps(dict(reference_sha256=sha(T/'tests/reference.py'),description='Full Fock/Bogoliubov channels with independently integrated flip noise and stationary joint spin/work jump-Hamiltonian response; prior heat references retained.',true_parameter=45.,groups={n:dict(inputs=hidden[n],truth=v.tolist()) for n,v in truth.items()}),indent=2)+'\n')
    frozen=json.loads(frozen_path.read_text());assert frozen['reference_sha256']==sha(T/'tests/reference.py') and all(rel(truth[n],np.array(frozen['groups'][n]['truth']))<1e-12 for n in hidden)
    gates=[n for n in hidden if n.startswith('gate_')]
    def scores(m,g):return {n:rel(m.predict_at(es,g),truth[n]) for n,es in hidden.items()}
    controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.Model().fit(records);res=(fit.predict(inputs)-y)/sigma;start=time.monotonic();row=dict(parameter=fit.conductance,calibration_chi2=float(res@res)/(len(records)-1),hidden=scores(m,fit.conductance),seconds=time.monotonic()-start)
        assert abs(fit.conductance/45.-1)<.03 and row['calibration_chi2']<1.5 and row['seconds']<45
        if name=='oracle':assert max(row['hidden'].values())<.04,row
        else:assert min(row['hidden'][n] for n in gates)>.04,row
        controls[name]=row
    print('CONTROLS',json.dumps(controls),flush=True)
    rng=np.random.default_rng(2026101120);domain=[]
    for i in range(32):
        left,right=rng.uniform(.25,1.05,2);phase=float(rng.uniform(-np.pi,np.pi));amplitude=float(rng.uniform(.3,1.));drive=float(rng.uniform(*((.1,.5) if i%2 else (7.,12.))));rho=float(rng.uniform(0,1));temperature=float(rng.uniform(12,24));secondary=float(rng.uniform(0,1));quadratic=float(rng.uniform(-.2,.2));momentum=float(rng.uniform(-.6,.6));conductance=float(rng.uniform(20,80))
        if i<8:left=(.25,1.05)[i&1];right=(.25,1.05)[(i>>1)&1];amplitude=(.3,1.)[(i>>2)&1];drive=(.1,.5,7.,12.)[i%4];rho=(0.,1.)[i&1];temperature=(12.,24.)[(i>>1)&1];secondary=(0.,1.)[(i>>2)&1];quadratic=(-.2,.2)[i&1];momentum=(-.6,.6)[(i>>1)&1];conductance=(20.,80.)[i&1]
        e=ref.gate_experiment(left,right,phase,amplitude,drive,rho,temperature,secondary,quadratic,momentum)
        a=good.gate_statistics(e,conductance);b=ref.gate_noise(e,conductance);fine=good.gate_statistics(e,conductance,nodes=288,count=80)
        row=dict(input=e,conductance=conductance,noise=a['noise'],shot=a['shot'],independent_noise=b,independent_error=rel(np.array(a['noise']),np.array(b)),refinement_error=rel(np.array(a['noise']),np.array(fine['noise'])),work_spin_identity_error=a['work_spin_identity_error'])
        assert a['noise']>0 and a['shot']>0 and row['independent_error']<3e-6 and row['refinement_error']<3e-6 and row['work_spin_identity_error']<1e-12,row
        if i in [0,7,15,31]:
            higher=ref.gate_noise(e,conductance,nodes=384,count=80,derivative_step=2e-6);row['reference_refinement_error']=rel(np.array(a['noise']),np.array(higher));assert row['reference_refinement_error']<3e-6,row
        # Work quanta equal drive/2 times injected total spin, so with no spin
        # loss the integrated work is bounded by the spin endpoint difference.
        state=good.stationary(*good.oscillator_arguments(e),conductance,secondary_coupling=secondary,quadratic_coupling=quadratic,momentum_coupling=momentum)
        ratio=good.QUASIPARTICLE_DENSITY/(good.NORMAL_POLARIZATION_TIME*good.RATE_CONVERSION*conductance)
        deltas=[good.gap(left,1.2),good.gap(right,1.9)];slopes=[good.spin_loss(t,d,m)[1] for t,d,m in zip([left,right],deltas,state['mu'])]
        tunnel_J=np.array(a['spin_jacobian'])+ratio*np.diag(slopes);no_loss_response=np.array(a['heat_derivative'])@np.linalg.inv(tunnel_J)
        row['no_spin_loss_conservation_error']=float(max(abs(no_loss_response-drive/2)));assert row['no_spin_loss_conservation_error']<1e-8,row
        domain.append(row)
        good.prepare.cache_clear();bad.prepare.cache_clear();ref.fock_channels.cache_clear()
        if (i+1)%8==0:print('Noise domain',i+1,'of32',round(time.monotonic()-tick,1),flush=True)
    limits=[]
    e=ref.gate_experiment(amplitude=0.)
    assert good.gate_statistics(e,45.)['noise']==bad.gate_statistics(e,45.)['noise']==ref.gate_noise(e,45.)==0.
    limits.append(dict(kind='static_barrier',noise=0.))
    e=ref.gate_experiment(amplitude=1.,drive=9.)
    old_tau=good.NORMAL_POLARIZATION_TIME
    try:
        good.NORMAL_POLARIZATION_TIME=old_tau/1e6;good.stationary.cache_clear()
        fast=good.gate_statistics(e,45.);difference=abs(fast['noise']/fast['shot']-1);assert difference<1e-5,difference
        limits.append(dict(kind='rapid_spin_relaxation',noise=fast['noise'],shot=fast['shot'],relative_error=difference,mu=fast['mu']))
    finally:
        good.NORMAL_POLARIZATION_TIME=old_tau;good.stationary.cache_clear()
    # Near-equilibrium work fluctuation-dissipation relation, with every bath
    # at a common temperature. This is an analytic limiting check outside the
    # high oscillator-temperature grading preparations.
    for drive in [.001,.002]:
        e=ref.gate_experiment(left=.8,right=.8,amplitude=.8,drive=drive,temperature=.8)
        a=good.gate_statistics(e,45.);state=good.stationary(*good.oscillator_arguments(e),45.,secondary_coupling=e['secondary_coupling'],quadratic_coupling=e['quadratic_coupling'],momentum_coupling=e['momentum_coupling'])
        fdt=2*good.KB*.8*state['heat'][2]*1e12;difference=abs(a['noise']/fdt-1);assert difference<2e-4,(drive,difference,a,fdt)
        limits.append(dict(kind='work_fluctuation_dissipation',drive=drive,noise=a['noise'],fdt=fdt,relative_error=difference))
    clean=good.predict_at(inputs,45.);rng=np.random.default_rng(43023);fits=[]
    for _ in range(256):
        noisy=clean+rng.normal(size=len(records))*sigma;sample=[dict(r,value=float(v)) for r,v in zip(records,noisy)];a=good.Model().fit(sample);b=bad.Model().fit(sample);res=(a.predict(inputs)-noisy)/sigma;chi=float(res@res)/(len(records)-1)
        assert abs(a.conductance/45.-1)<.03 and abs(a.conductance-b.conductance)<1e-10 and chi<1.5;fits.append(dict(parameter=a.conductance,chi2=chi))
    params=np.array([r['parameter'] for r in fits]);sensitivity=[]
    for g in np.quantile(params,np.linspace(0,1,9)):
        a=scores(good,g);b=scores(bad,g);assert max(a.values())<.04 and min(b[n] for n in gates)>.04;sensitivity.append(dict(parameter=float(g),oracle=a,shortcut=b));print('Noise fits',len(sensitivity),'of9',flush=True)
    report=dict(revision=10,calibration_regenerated=False,inherited_r9_validation_sha256=sha(A/'validation.json'),controls=controls,physics=dict(domain=domain,limits=limits),noise=dict(seed=43023,trials=256,method='256 actual calibration fits; all hidden predictions at nine fitted-parameter quantiles including extrema.',fit_min=float(params.min()),fit_max=float(params.max()),max_chi2=max(r['chi2'] for r in fits),hidden_sensitivity=sensitivity),seconds=time.monotonic()-tick,source_sha256={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and not any(k in p.parts for k in ['__pycache__','.pytest_cache'])})
    (S/'results/superconducting-heat-r10-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',round(report['seconds'],1),flush=True)
if __name__=='__main__':main()
