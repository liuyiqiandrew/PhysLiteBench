"""Validate thermal Fock phase coherence, stationary spin balance and calorimetry."""
import argparse,hashlib,importlib.util,json,shutil,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.stats import poisson

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/superconducting-heat'


def module(path):
    spec=importlib.util.spec_from_file_location('science_'+str(abs(hash(str(path)))),path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def local_control(source):
    start=time.time()
    with tempfile.TemporaryDirectory(prefix='physlite-oscillator-control-') as temporary:
        app,tests=Path(temporary)/'app',Path(temporary)/'tests'
        shutil.copytree(TASK/'environment',app);shutil.copytree(TASK/'tests',tests)
        shutil.copyfile(source,app/'model.py')
        value=subprocess.run([sys.executable,'-m','pytest','-q',str(app/'test_public.py'),str(tests/'test_hidden.py')],cwd=app,text=True,capture_output=True,env=dict(__import__('os').environ,PYTHONPATH=str(app)))
        return dict(returncode=value.returncode,stdout=value.stdout,stderr=value.stderr,seconds=time.time()-start)


def phase_space(x,t):
    if x==0:return t
    if x/t>40:return x
    if x/t< -40:return -x*np.exp(x/t)
    return x/(-np.expm1(-x/t))


def normal_closed(good,t,a,w,rho,tph,g):
    ells,prob=good.sidebands(rho,tph,32)
    def current(v):
        return sum(p*(phase_space(v-ell*24,t)-phase_space(-v-ell*24,t)) for ell,p in zip(ells,prob))
    ratio=good.QUASIPARTICLE_DENSITY/(good.NORMAL_POLARIZATION_TIME*good.RATE_CONVERSION*g)
    return brentq(lambda mu:a*a/2*current(w-2*mu)-ratio*mu,0.,w/2,xtol=5e-15)


def spin_basis_check(ref):
    sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]],complex);sz=np.diag([1.,-1.]);identity=np.eye(2)
    axis=(sx+2*sy+3*sz)/np.sqrt(14)
    fl=np.diag([.3,.08]);fr=np.diag([.7,.16])
    def rates(matrix,left,right,z,pair,pf,pb):
        if pair:
            l=pf*(identity-left)@matrix@(identity-right).T@matrix.conj().T-pb*left@matrix@right.T@matrix.conj().T
            r=pf*(identity-right)@matrix.T@(identity-left).T@matrix.conj()-pb*right@matrix.T@left.T@matrix.conj()
            return np.array([np.trace(l).real,np.trace(z@l).real,np.trace(z@r).real])
        l=pf*left@matrix@(identity-right)@matrix.conj().T-pb*(identity-left)@matrix@right@matrix.conj().T
        r=pf*(identity-right)@matrix.conj().T@left@matrix-pb*right@matrix.conj().T@(identity-left)@matrix
        return np.array([np.trace(l).real,-np.trace(z@l).real,np.trace(z@r).real])
    error=0.
    for harmonic in (-1,0,1):
        table=ref.matrices(.8);t=table[harmonic];h=ref.S@table[-harmonic].conj()@ref.S.conj().T
        for ell in (0,1,2):
            plus=(1j)**ell;minus=(-1j)**ell
            for pair in (False,True):
                matrix=(.8*.6*plus*t+.6*.8*np.exp(.7j)*minus*h)@ref.S if pair else .8*.8*plus*t-.6*.6*np.exp(.7j)*minus*h
                original=rates(matrix,fl,fr,sz,pair,.7,.2)
                for angle in (.4,1.7,2.8):
                    u=np.cos(angle/2)*identity-1j*np.sin(angle/2)*axis
                    changed=u.conj().T@matrix@(u.conj() if pair else u)
                    actual=rates(changed,u.conj().T@fl@u,u.conj().T@fr@u,u.conj().T@sz@u,pair,.7,.2)
                    error=max(error,float(max(abs(actual-original))))
    return error


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    start=time.time();good,bad,ref=[module(p) for p in (TASK/'solution/model.py',ROOT/'scripts/superconducting_heat_baseline.py',TASK/'tests/reference.py')]
    old=module(ROOT/'archives/superconducting-heat-r4/tasks/superconducting-heat/solution/model.py')
    meta=json.loads((TASK/'tests/metadata.json').read_text());true,sigma,limit=ref.TRUE_PARAMETER,meta['measurement_sigma'],meta['prediction_limit']
    records=json.loads((TASK/'environment/data/calibration.json').read_text());inputs=ref.calibration_inputs()
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(ROOT/'archives/superconducting-heat-r4/tasks/superconducting-heat/environment/data/calibration.json').read_bytes()
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    exact=ref.predict(inputs,true);hidden=ref.hidden_inputs();truth={name:ref.predict(es,true) for name,es in hidden.items()}
    def chi(model,rs):
        residual=(model.predict(inputs)-np.array([r['value'] for r in rs]))/sigma
        return float(residual@residual/(len(rs)-1))
    def scores(model):
        return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2)/np.mean(truth[name]**2))) for name,es in hidden.items()}
    report=dict(revision=6,calibration_preserved=True,noise_trials=args.noise_trials,measurement_sigma_pW=sigma,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],controls={})
    for label,source in (('oracle',good),('shortcut',bad)):
        model=source.Model().fit(records)
        result=dict(parameter=model.conductance,parameter_relative_error=abs(model.conductance/true-1),calibration_chi2=chi(model,records),hidden=scores(model))
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert max(result['hidden'].values())<limit if label=='oracle' else min(result['hidden'].values())>limit
        report['controls'][label]=result
    print('Frozen controls passed',flush=True)
    rng=np.random.default_rng(meta['noise_seed']);parameters=[];chi2s=[];oracle_errors=[];source_errors=[]
    for trial in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a,b=good.Model().fit(sample),bad.Model().fit(sample)
        assert abs(a.conductance-b.conductance)<1e-10
        parameters.append(a.conductance);chi2s.append(chi(a,sample));oracle_errors.extend(scores(a).values());source_errors.extend(scores(b).values())
        if (trial+1)%32==0:print('Noise draws',trial+1,flush=True)
    assert max(chi2s)<1.5 and max(abs(np.array(parameters)/true-1))<.03
    assert max(oracle_errors)<limit and min(source_errors)>limit
    report['noise']=dict(parameter_min=min(parameters),parameter_max=max(parameters),parameter_relative_error_max=float(max(abs(np.array(parameters)/true-1))),calibration_chi2_max=max(chi2s),oracle_prediction_error_max=max(oracle_errors),shortcut_prediction_error_min=min(source_errors),oracle_passes=args.noise_trials,shortcut_intended_failures=args.noise_trials)
    print('Noise checks passed',flush=True)
    corners=[]
    for j in range(64):
        corners.append(((.25,1.05)[j&1],(.25,1.05)[(j>>1)&1],(0.,np.pi)[j%2],(0.,1.)[(j>>2)&1],(.1,.5,7.,12.)[(j>>3)&3],(0.,1.)[(j>>5)&1],(12.,24.)[(j>>4)&1]))
    rng=np.random.default_rng(931006)
    random=[]
    for i in range(32):
        tl,tr=rng.uniform(.25,1.05,2);random.append((tl,tr,rng.uniform(-np.pi,np.pi),rng.uniform(0,1),rng.uniform(*((.1,.5) if i%2 else (7,12))),rng.uniform(0,1),rng.uniform(12,24)))
    cases=corners+random
    checks=[]
    for j,case in enumerate(cases):
        a=good.stationary(*case,true);b=bad.stationary(*case,true);r=ref.oscillator_components(*case,true)
        finer=good.stationary(*case,true,nodes=288,count=32)
        heat,spin=a['heat'],a['spin'];jac=a['jacobian'];ells,prob=good.sidebands(case[5],case[6],24);fock=ref.fock_covariances(case[5],case[6],24)
        nb=1/np.expm1(24/case[6]);dw=np.exp(-2*case[5]*(2*nb+1))
        ent=a['entropy']+a['flip_entropy'];sent=b['entropy']+b['flip_entropy'];thermo=(-heat[0]/case[0]-heat[1]/case[1]+heat[3]/case[6])*1e-12
        reverse=good.stationary(case[0],case[1],-case[2],*case[3:],true)
        row=dict(input=case,polarization_K=a['mu'].tolist(),heat_pW=heat.tolist(),source_heat_pW=b['heat'].tolist(),entropy_W_per_K=float(ent),source_entropy_W_per_K=float(sent),independent_heat_max_pW=float(max(abs(heat-r['heat']))),independent_mu_max_K=float(max(abs(a['mu']-r['mu']))),refined_heat_max_pW=float(max(abs(heat-finer['heat']))),refined_mu_max_K=float(max(abs(a['mu']-finer['mu']))),energy_identity_pW=float(heat[0]+heat[1]+heat[2]-heat[3]),spin_work_identity_pW=float(heat[2]-good.KB*case[4]/2*sum(spin)*1e12),entropy_identity_W_per_K=float(ent-thermo),minimum_rate=a['minimum_rate'],source_minimum_rate=b['minimum_rate'],balance_residual_normalized=float(max(abs(a['residual']))),jacobian_column_dominance_min=float(min(-np.diag(jac)-np.sum(abs(jac),axis=0)+abs(np.diag(jac)))),phase_reversal_pW=float(max(abs(heat-reverse['heat']))),probability_sum_error=float(sum(prob)-1),mean_recoil_error=float(ells@prob-case[5]),recoil_variance_error=float((ells-case[5])**2@prob-case[5]*(2*nb+1)),anomalous_weight_sum_error=float(((-1.)**ells)@prob-dw),fock_probability_error=float(max(abs(prob-np.array([v[1] for v in fock])))),fock_cross_error=float(max(abs(((-1.)**ells)*prob-np.array([v[3] for v in fock])))),search_bound=a['search_bound'],search_expansions=a['expansions'])
        assert row['independent_heat_max_pW']<1e-7 and row['refined_heat_max_pW']<1e-7
        assert row['independent_mu_max_K']<1e-7 and row['refined_mu_max_K']<1e-7
        assert abs(row['energy_identity_pW'])<1e-9 and abs(row['spin_work_identity_pW'])<1e-9
        assert ent>=-1e-24 and sent>=-1e-24 and abs(row['entropy_identity_W_per_K'])<1e-21
        assert row['minimum_rate']>=-1e-20 and row['source_minimum_rate']>=-1e-20
        assert row['balance_residual_normalized']<1e-9 and row['jacobian_column_dominance_min']>0
        assert row['phase_reversal_pW']<1e-8
        for key in ('probability_sum_error','mean_recoil_error','recoil_variance_error','anomalous_weight_sum_error','fock_probability_error','fock_cross_error'):assert abs(row[key])<1e-12
        checks.append(row)
        if (j+1)%16==0:print('Domain systems',j+1,flush=True)
    rhozero=[];normal=[];extra=[];multiple=[];refinement=[];equilibrium=[];contrast=[]
    for case in random[:12]:
        args0=(*case[:5],0.,case[6]);a=good.stationary(*args0,true);oldstate=old.stationary_state(*case[:5],true)
        rhozero.append(float(max(abs(a['heat'][:3]-oldstate['heat']))))
        base=good.stationary(*case,true)
        for seed in ((5.,5.),(-5.,-5.),(5.,-5.)):
            other=good.stationary(*case,true,initial=seed);multiple.append(float(max(abs(base['mu']-other['mu']))))
        for conductance in (20.,80.):
            a=good.stationary(*case,conductance);r=ref.oscillator_components(*case,conductance);extra.append(float(max(abs(a['heat']-r['heat']))))
        other=ref.oscillator_components(*case,true,nodes=384,count=32);refinement.append(float(max(abs(ref.oscillator_components(*case,true)['heat']-other['heat']))))
    for t in (.25,1.05):
        for w in (.3,9.):
            case=(t,t,.7,.8,w,.7,24.);a=good.stationary(*case,true,normal=True);b=bad.stationary(*case,true,normal=True);r=ref.oscillator_components(*case,true,normal=True);mu=normal_closed(good,t,.8,w,.7,24.,true)
            normal.append(dict(input=case,closed_mu_K=mu,closed_mu_error_K=float(max(abs(a['mu']-mu))),independent_heat_error_pW=float(max(abs(a['heat']-r['heat']))),shortcut_normal_heat_error_pW=float(max(abs(a['heat']-b['heat'])))))
    for drive in (.3,9.):
        for amplitude in (0.,.8):
            case=(.8,.8,.7,amplitude,drive,.7,.8);a=good.stationary(*case,true)
            equilibrium.append(dict(input=case,work_pW=float(a['heat'][2]),entropy_W_per_K=float(a['entropy']+a['flip_entropy']),spin_balance=float(max(abs(a['residual'])))))
            assert a['heat'][2]>=-1e-10 and equilibrium[-1]['entropy_W_per_K']>=-1e-24
    for es in hidden.values():
        for e in es:
            args=good.oscillator_arguments(e);other=(*args[:2],(args[2]+2*np.pi)%(2*np.pi)-np.pi,*args[3:]);one,two=good.stationary(*args,true),good.stationary(*other,true)
            value=good.predict_at([e],true)[0];contrast.append(dict(input=e,readout_identity_pW=float(value-(one['heat'][0]-two['heat'][0])/2),phase_population_difference_K=float(max(abs(one['mu']-two['mu'])))))
    assert max(rhozero)<1e-7 and max(multiple)<1e-7 and max(extra)<1e-7 and max(refinement)<1e-7
    assert max(r['closed_mu_error_K'] for r in normal)<1e-9
    assert max(r['independent_heat_error_pW'] for r in normal)<1e-7 and max(r['shortcut_normal_heat_error_pW'] for r in normal)<1e-9
    assert max(abs(r['readout_identity_pW']) for r in contrast)<1e-10
    basis=spin_basis_check(ref);assert basis<1e-12
    largest_total=2/np.expm1(24/24)+1
    tail=float(poisson.sf(24,largest_total));second_moment_tail=float(largest_total**2*poisson.sf(22,largest_total)+largest_total*poisson.sf(23,largest_total))
    assert tail<1e-16 and second_moment_tail<1e-13
    report['physics']=dict(domain_seed=931006,domain_systems=len(checks),domain=checks,rho_zero_previous_heat_error_pW=max(rhozero),normal_state=normal,extra_conductance_agreement_max_pW=max(extra),multiple_start_mu_max_K=max(multiple),independent288_vs384_24_vs32_max_pW=max(refinement),common_temperature_passivity=equilibrium,contrast_preparation=contrast,spin_basis_invariance=basis,oscillator_sideband_probability_tail_bound=tail,oscillator_second_recoil_moment_tail_bound=second_moment_tail,initial_Fock_state_tail_bound=float(np.exp(-96)),tail_derivation='Absolute sideband gain is bounded by total emission+absorption count, Poisson with max mean rho(2nB+1); unitarity bounds discarded initial thermal Fock probability.',independent_reference='Explicit thermal Laguerre/Fock displacement covariances, four normal-dispersion Bogoliubov branches, 2x2 singlet spin matrices, independently checked expanding-window root balance and higher-order quadrature.')
    local={label:local_control(source) for label,source in (('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/superconducting_heat_baseline.py'))}
    assert local['oracle']['returncode']==0 and local['oracle']['seconds']<60
    assert local['shortcut']['returncode']!=0 and local['shortcut']['seconds']<60
    assert '3 failed, 4 passed' in local['shortcut']['stdout']
    report['local_controls']=local;report['seconds']=time.time()-start
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[ROOT/'scripts/superconducting_heat_baseline.py',Path(__file__)]
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    output=ROOT/'results/superconducting-heat-r6-validation.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('physics','source_sha256','local_controls')},indent=2))
    print('Full scientific report:',str(output),flush=True)


if __name__=='__main__':main()
