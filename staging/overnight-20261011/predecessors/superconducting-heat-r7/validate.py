"""Validate commensurate thermal-mode coherence and self-consistent calorimetry."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,platform,time
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.stats import poisson

ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/superconducting-heat'
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def error(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def phase_space(x,t):
    if abs(x)<1e-12:return t
    if x/t>40:return x
    if x/t<-40:return -x*np.exp(x/t)
    return x/(-np.expm1(-x/t))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args();start=time.monotonic()
    good=load('oracle',TASK/'solution/model.py');bad=load('shortcut',ROOT/'scripts/superconducting_heat_baseline.py');ref=load('reference',TASK/'tests/reference.py')
    prior=load('previous',ROOT.parents[1]/'tasks/superconducting-heat/solution/model.py')
    rs=json.loads((TASK/'environment/data/calibration.json').read_text());assert rs==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert sha(TASK/'environment/data/calibration.json')==sha(ROOT.parents[1]/'tasks/superconducting-heat/environment/data/calibration.json')
    assert 'coherences' not in (TASK/'environment/model.py').read_text() and 'signed=np' not in (TASK/'environment/model.py').read_text()
    inputs=[r['input'] for r in rs];y=np.array([r['value'] for r in rs]);sigma=np.array([r['sigma'] for r in rs]);hidden=ref.hidden_inputs()
    truth={}
    for name,es in hidden.items():
        truth[name]=ref.predict(es,45.);print('Independent hidden group',name,round(time.monotonic()-start,1),flush=True)
    frozen_path=TASK/'tests/data/hidden_truth.json'
    if args.freeze:frozen_path.write_text(json.dumps(dict(description='Independent products of direct thermal Fock displacement covariances and spin-matrix transition rates.',reference_sha256=sha(TASK/'tests/reference.py'),true_parameter=45.,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,machine=platform.machine()),groups={n:dict(inputs=hidden[n],truth=v.tolist()) for n,v in truth.items()}),indent=2)+'\n')
    frozen=json.loads(frozen_path.read_text());assert frozen['reference_sha256']==sha(TASK/'tests/reference.py')
    assert all(error(np.array(frozen['groups'][n]['truth']),truth[n])<1e-9 for n in hidden)
    def scores(m,g):return {n:error(m.predict_at(es,g),truth[n]) for n,es in hidden.items()}
    controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.Model().fit(rs);res=(fit.predict(inputs)-y)/sigma;tick=time.monotonic()
        row=dict(parameter=fit.conductance,calibration_chi2=float(res@res)/(len(rs)-1),hidden=scores(m,fit.conductance),hidden_seconds=time.monotonic()-tick)
        assert abs(fit.conductance/45-1)<.03 and row['calibration_chi2']<1.5
        assert max(row['hidden'].values())<.04 if name=='oracle' else min(row['hidden'].values())>.04
        assert row['hidden_seconds']<45,row
        controls[name]=row
    print('Controls',json.dumps(controls),flush=True)
    rng=np.random.default_rng(202610117);spectra=[]
    for i in range(64):
        rho,secondary=rng.uniform(0,1,2);temperature=float(rng.uniform(12,24))
        if i<8:rho=float(i&1);secondary=float((i>>1)&1);temperature=(12.,24.)[(i>>2)&1]
        ell,p,cross=good.bath_spectrum(float(rho),float(secondary),temperature)
        fock=ref.two_mode_fock_covariances(float(rho),float(secondary),temperature)
        fp=np.array([r[1] for r in fock]);fc=np.array([r[3].real for r in fock])
        nb1=1/np.expm1(24/temperature);nb2=1/np.expm1(48/temperature)
        mean=rho+2*secondary;variance=rho*(2*nb1+1)+4*secondary*(2*nb2+1)
        dw=np.exp(-2*rho*(2*nb1+1)-2*secondary*(2*nb2+1))
        positive=ell>=0
        row=dict(rho=rho,secondary=secondary,temperature=temperature,probability_error=float(abs(p.sum()-1)),mean_error=float(abs(ell@p-mean)),variance_error=float(abs((ell-mean)**2@p-variance)),coherence_sum_error=float(abs(cross.sum()-dw)),fock_probability_error=float(max(abs(p-fp))),fock_cross_error=float(max(abs(cross-fc))),minimum_covariance_eigenvalue=float(min(p-abs(cross))),kms_error=float(max(abs(p[::-1][positive]-p[positive]*np.exp(-ell[positive]*24/temperature)))),anomalous_kms_error=float(max(abs(cross[::-1][positive]-cross[positive]*np.exp(-ell[positive]*24/temperature)))))
        assert max(abs(v) for k,v in row.items() if k.endswith('error'))<2e-12,row
        assert row['minimum_covariance_eigenvalue']>-1e-15
        spectra.append(row)
    print('Independent Fock spectra and thermal identities PASS',flush=True)
    cases=[]
    for i in range(16):
        cases.append(((.25,1.05)[i&1],(.25,1.05)[(i>>1)&1],(0.,np.pi)[i&1],(0.,1.)[(i>>2)&1],(.1,.5,7.,12.)[(i>>2)&3],(0.,1.)[(i>>3)&1],(12.,24.)[(i>>1)&1],(0.,1.)[i&1]))
    for i in range(16):cases.append((*rng.uniform(.25,1.05,2),float(rng.uniform(-np.pi,np.pi)),float(rng.uniform(0,1)),float(rng.uniform(*((.1,.5) if i%2 else (7.,12.)))),float(rng.uniform(0,1)),float(rng.uniform(12,24)),float(rng.uniform(0,1))))
    domain=[]
    for i,case in enumerate(cases):
        settings=case[:7];secondary=case[7];conductance=[20.,45.,80.][i%3]
        a=good.stationary(*settings,conductance,secondary_coupling=secondary);b=bad.stationary(*settings,conductance,secondary_coupling=secondary)
        independent=ref.oscillator_components(*settings,conductance,secondary_coupling=secondary)
        finer=good.stationary(*settings,conductance,nodes=288,count=32,secondary_coupling=secondary)
        heat=a['heat'];jac=a['jacobian'];ent=a['entropy']+a['flip_entropy'];sent=b['entropy']+b['flip_entropy']
        thermo=(-heat[0]/case[0]-heat[1]/case[1]+heat[3]/case[6])*1e-12
        reverse=good.stationary(*settings[:2],-settings[2],*settings[3:],conductance,secondary_coupling=secondary)
        row=dict(input=case,conductance=conductance,independent_heat_error=float(max(abs(heat-independent['heat']))),independent_mu_error=float(max(abs(a['mu']-independent['mu']))),refinement_heat_error=float(max(abs(heat-finer['heat']))),first_law_error=float(abs(heat[0]+heat[1]+heat[2]-heat[3])),spin_work_error=float(abs(heat[2]-good.KB*case[4]/2*sum(a['spin'])*1e12)),entropy=float(ent),shortcut_entropy=float(sent),entropy_identity_error=float(abs(ent-thermo)),minimum_rate=float(a['minimum_rate']),shortcut_minimum_rate=float(b['minimum_rate']),balance_residual=float(max(abs(a['residual']))),jacobian_column_dominance=float(min(-np.diag(jac)-np.sum(abs(jac),axis=0)+abs(np.diag(jac)))),phase_reversal_error=float(max(abs(heat-reverse['heat']))))
        assert max(row[k] for k in ['independent_heat_error','independent_mu_error','refinement_heat_error','phase_reversal_error'])<1e-7,row
        assert row['first_law_error']<1e-9 and row['spin_work_error']<1e-9 and row['entropy_identity_error']<1e-21
        assert ent>=-1e-24 and sent>=-1e-24 and row['minimum_rate']>=-1e-19 and row['shortcut_minimum_rate']>=-1e-19
        assert row['balance_residual']<1e-9 and row['jacobian_column_dominance']>0
        if i<4:
            for initial in [(5.,5.),(-5.,-5.),(5.,-5.)]:
                other=good.stationary(*settings,conductance,secondary_coupling=secondary,initial=initial);assert max(abs(a['mu']-other['mu']))<1e-7
        domain.append(row)
        if (i+1)%8==0:print('Full physics domain systems',i+1,round(time.monotonic()-start,1),flush=True)
        # Bound retained channel memory while keeping enough reuse for the noise check.
        good.prepare.cache_clear();bad.prepare.cache_clear();ref.fock_channels.cache_clear()
    limits=[]
    for temperature in [.25,1.05]:
        for drive in [.3,9.]:
            settings=(temperature,temperature,.7,.8,drive,.7,24.);secondary=.9
            a=good.stationary(*settings,45.,normal=True,secondary_coupling=secondary);b=bad.stationary(*settings,45.,normal=True,secondary_coupling=secondary)
            ell,p,_=good.bath_spectrum(.7,secondary,24.,32)
            def current(v):return sum(prob*(phase_space(v-k*24,temperature)-phase_space(-v-k*24,temperature)) for k,prob in zip(ell,p))
            ratio=good.QUASIPARTICLE_DENSITY/(good.NORMAL_POLARIZATION_TIME*good.RATE_CONVERSION*45.)
            mu=brentq(lambda value:.8**2/2*current(drive-2*value)-ratio*value,0.,drive/2,xtol=5e-15)
            assert max(abs(a['mu']-mu))<1e-9 and max(abs(a['heat']-b['heat']))<1e-8
            limits.append(dict(temperature=temperature,drive=drive,normal_closed_mu=mu,mu_error=float(max(abs(a['mu']-mu)))))
    for case in cases[-4:]:
        settings=case[:7];new=good.stationary(*settings,45.,secondary_coupling=0.)['heat'];old=prior.stationary(*settings,45.)['heat'];assert max(abs(new-old))<1e-9
        e=dict(left_temperature=case[0],right_temperature=case[1],phase=case[2],amplitude=case[3],drive_energy=case[4])
        assert abs(good.predict_at([e],45.)[0]-prior.predict_at([e],45.)[0])<1e-9
    for amplitude in [0.,.8]:
        eq=good.stationary(.8,.8,.7,amplitude,9.,.7,.8,45.,secondary_coupling=.9)
        assert eq['heat'][2]>=-1e-10 and eq['entropy']+eq['flip_entropy']>=-1e-24
        if amplitude==0:assert max(abs(eq['heat']))<1e-9
    print('Normal-state, single-mode, common-temperature limits PASS',flush=True)
    clean=good.predict_at(inputs,45.);rng=np.random.default_rng(43021);fits=[]
    for _ in range(256):
        ys=clean+rng.normal(size=len(rs))*sigma;sample=[dict(r,value=float(v)) for r,v in zip(rs,ys)];a=good.Model().fit(sample);b=bad.Model().fit(sample)
        res=(a.predict(inputs)-ys)/sigma;chi=float(res@res)/(len(rs)-1)
        assert abs(a.conductance-b.conductance)<1e-10 and abs(a.conductance/45-1)<.03 and chi<1.5
        fits.append(dict(parameter=a.conductance,chi2=chi))
    parameters=np.array([r['parameter'] for r in fits]);sensitivity=[]
    for g in np.quantile(parameters,np.linspace(0,1,9)):
        oe=scores(good,float(g));be=scores(bad,float(g));assert max(oe.values())<.04 and min(be.values())>.04
        sensitivity.append(dict(conductance=float(g),oracle=oe,shortcut=be))
        print('Noisy-fit sensitivity',len(sensitivity),'of9',flush=True)
    tail1=poisson.sf(24,1+2/np.expm1(24/24));tail2=poisson.sf(24,1+2/np.expm1(48/24))
    report=dict(revision=7,calibration_regenerated=False,controls=controls,physics=dict(spectra=spectra,domain=domain,limits=limits,sideband_tail_bound=float(tail1+tail2),initial_fock_tail_bound=float(np.exp(-96)+np.exp(-192))),noise=dict(seed=43021,trials=256,method='256 actual calibration fits; all hidden predictions evaluated at nine fitted-parameter quantiles including both extrema. Not 256 full hidden predictions.',fit_min=float(parameters.min()),fit_max=float(parameters.max()),max_calibration_chi2=max(r['chi2'] for r in fits),hidden_sensitivity=sensitivity,oracle_worst=max(max(r['oracle'].values()) for r in sensitivity),shortcut_smallest_miss=min(min(r['shortcut'].values()) for r in sensitivity)),seconds=time.monotonic()-start,source_sha256={str(p.relative_to(TASK)):sha(p) for p in TASK.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache'])})
    (ROOT/'results/superconducting-heat-r7-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',seconds=report['seconds'],oracle_worst=report['noise']['oracle_worst'],shortcut_smallest_miss=report['noise']['shortcut_smallest_miss'])),flush=True)
if __name__=='__main__':main()
