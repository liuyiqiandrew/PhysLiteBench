"""Scientific calibration, conservation and independent-grid checks; no model runs."""
from pathlib import Path
import argparse,hashlib,importlib.util,itertools,json,time
import numpy as np

BASE=Path(__file__).resolve().parents[1]
TASK=BASE/'tasks/acoustic-cavity-pressure'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def normalized(actual,truth):
    return float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    oracle=load(TASK/'solution/model.py','oracle')
    shortcut=load(BASE/'scripts/acoustic_cavity_pressure_baseline.py','shortcut')
    reference=load(TASK/'tests/reference.py','reference')
    meta=json.loads((TASK/'tests/metadata.json').read_text());sigma=meta['sigma'];true=reference.TRUE_PARAMETER
    inputs=reference.calibration_inputs();clean=reference.predict(inputs)
    if args.generate:
        rng=np.random.default_rng(meta['calibration_seed'])
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,clean+rng.normal(0,sigma,len(inputs)))]
        text=json.dumps(records,indent=2)+'\n'
        for p in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:p.write_text(text)
    records=json.loads((TASK/'tests/data/calibration.json').read_text())
    assert records==json.loads((TASK/'environment/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==sigma for r in records)
    assert meta['prediction_limit']==.04 and sigma==.0002
    hidden=reference.hidden_inputs();truth={k:reference.predict(es) for k,es in hidden.items()}
    groups=[k for k in hidden if k!='density_anchors']
    def evaluate(module,rr):
        m=module.Model().fit(rr);y=np.array([r['value'] for r in rr])
        return {'viscosity':m.viscosity,'parameter_relative_error':abs(m.viscosity/true-1),
                'calibration_chi2':float(np.sum(((m.predict(inputs)-y)/sigma)**2)/(len(inputs)-1)),
                'hidden':{k:normalized(m.predict(es),truth[k]) for k,es in hidden.items()}}
    nominal={name:evaluate(module,records) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    assert max(nominal['oracle']['hidden'].values())<.04
    assert min(nominal['shortcut']['hidden'][k] for k in groups)>.04
    noise=[];rng=np.random.default_rng(meta['noise_validation_seed'])
    for _ in range(256):
        rr=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,clean+rng.normal(0,sigma,len(inputs)))]
        results={name:evaluate(module,rr) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
        for res in results.values():assert res['parameter_relative_error']<.03 and res['calibration_chi2']<1.5
        assert max(results['oracle']['hidden'].values())<.04
        assert min(results['shortcut']['hidden'][k] for k in groups)>.04
        noise.append(results)
    # Exact monotonicity of each positive harmonic-amplitude calibration response.
    unique=[dict(t) for t in {tuple(sorted(e.items())) for e in inputs}]
    slopes=[];recover=[];profile=[]
    for eta in np.linspace(.08,.18,81):
        derivative=[]
        for e in unique:
            k=e['mode']*np.pi/e['length'];omega=e['frequency_ratio']*k
            detuning=k*k/omega-omega
            prefactor=k*e['drive']/omega
            derivative.append(-prefactor*eta*k**4/(eta**2*k**4+detuning**2)**1.5)
        slopes+=derivative
    assert max(slopes)<0
    for eta in [.08,.10,.12,.15,.18]:
        y=oracle.predict_at(inputs,eta)
        rr=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,y)]
        recover.append(abs(oracle.Model().fit(rr).viscosity/eta-1))
        grid=np.linspace(.08,.18,401);loss=[float(np.sum((oracle.predict_at(inputs,x)-y)**2)) for x in grid]
        profile.append({'truth':eta,'grid_minimum':float(grid[np.argmin(loss)]),'minimum_loss':min(loss)})
        assert abs(grid[np.argmin(loss)]-eta)<.000251
    assert max(recover)<1e-7
    cases=[(L,m,r,f,eta) for L,m,r,f,eta in itertools.product([2.5,4.],[1,2,3],[.8,1.,1.2],[.1,.3],[.08,.12,.18])]
    rng=np.random.default_rng(241047)
    cases += [(float(rng.uniform(2.5,4)),int(rng.integers(1,4)),float(rng.uniform(.8,1.2)),float(rng.uniform(.1,.3)),float(rng.uniform(.08,.18))) for _ in range(24)]
    comparisons=[];balances=[]
    for L,m,r,f,eta in cases:
        es=[reference.experiment(L,m,r,f,k) for k in ['density','force']]
        actual=oracle.predict_at(es,eta);independent=reference.predict(es,eta)
        refined=reference.predict(es,eta,cells=384)
        comparisons.append({'input':es[0],'viscosity':eta,
            'reference_relative_error':float(np.max(abs(actual/independent-1))),
            'refinement_relative_change':float(np.max(abs(refined/independent-1)))})
        balances.append(reference.finite_volume(L,m,r,f,eta,256))
    ref_error=max(x['reference_relative_error'] for x in comparisons)
    refinement=max(x['refinement_relative_change'] for x in comparisons)
    assert ref_error<1e-5 and refinement<1e-5
    assert max(abs(x['mean_density_integral']) for x in balances)<1e-11
    assert max(x['mean_mass_flux_max'] for x in balances)<1e-11
    assert max(x['mean_momentum_flux_variation'] for x in balances)<1e-10
    assert max(x['linear_energy_balance'] for x in balances)<1e-10
    assert max(x['mean_velocity_max'] for x in balances)<1e-10
    resonance=[];scaling=[];reflection=[]
    for L,m,eta in [(2.7,1,.08),(3.3,2,.12),(4.,3,.18)]:
        es=[reference.experiment(L,m,1.,.2)]
        resonance.append(abs(shortcut.predict_at(es,eta)[0]/oracle.predict_at(es,eta)[0]-4/3))
        low=[reference.experiment(L,m,.93,.1,k) for k in ['density','force']]
        high=[reference.experiment(L,m,.93,.2,k) for k in ['density','force']]
        for module in [oracle,shortcut]:scaling.append(float(np.max(abs(module.predict_at(high,eta)/module.predict_at(low,eta)-np.array([2.,4.])))))
        # A sine mode has equal density amplitude at both fixed walls, with parity sign.
        k,omega=m*np.pi/L,.93*m*np.pi/L
        v,d=oracle.fields(L,m,.93,.2,eta)
        reflection.append(abs(abs(d*np.cos(k*L))-abs(d)))
    assert max(resonance)<1e-12 and max(scaling)<1e-12 and max(reflection)<1e-12
    equivalence=float(np.max(abs(oracle.predict_at(inputs,true)-shortcut.predict_at(inputs,true))))
    assert equivalence==0
    report={'task':'acoustic-cavity-pressure','revision':1,'model_evaluations':0,'noise_trials':256,'controls':nominal,
        'noise':{'oracle_passes':256,'shortcut_rejections':256,'all_calibration_pass':True,
            'max_parameter_relative_error':max(x[n]['parameter_relative_error'] for x in noise for n in x),
            'max_calibration_chi2':max(x[n]['calibration_chi2'] for x in noise for n in x),
            'max_oracle_hidden_error':max(v for x in noise for v in x['oracle']['hidden'].values()),
            'min_shortcut_force_error':min(x['shortcut']['hidden'][k] for x in noise for k in groups)},
        'physics':{'independent_cases':len(cases),'oracle_reference_relative_error_max':ref_error,
            'reference_refinement_relative_change_max':refinement,
            'mean_mass_error_max':max(abs(x['mean_density_integral']) for x in balances),
            'mean_mass_flux_max':max(x['mean_mass_flux_max'] for x in balances),
            'mean_momentum_flux_variation_max':max(x['mean_momentum_flux_variation'] for x in balances),
            'mean_velocity_max':max(x['mean_velocity_max'] for x in balances),
            'drive_dissipation_balance_max':max(x['linear_energy_balance'] for x in balances),
            'resonance_ratio_error_max':max(resonance),'drive_scaling_error_max':max(scaling),
            'endwall_reflection_error_max':max(reflection),'calibration_equivalence_max':equivalence,
            'minimum_hidden_force_signal':min(float(np.min(abs(truth[k]))) for k in groups),
            'maximum_hidden_force_signal':max(float(np.max(abs(truth[k]))) for k in groups)},
        'identifiability':{'basis':'Every positive density amplitude is strictly decreasing in positive viscosity; its analytic derivative is checked across the bounded interval.',
            'minimum_negative_slope':float(-max(slopes)),'max_noiseless_relative_recovery_error':max(recover),'objective_profiles':profile},
        'data_integrity':{'records':len(records),'unique_settings':len(unique),'fixed_instrument_sigma':sigma,
            'public_private_identical':True,'sigma_independent_of_response_and_parameter':True,'calibration_seed':meta['calibration_seed'],'noise_seed':meta['noise_validation_seed']},
        'calibration_readout_note':'Optical density harmonic; not a mechanical pressure sensor, whose linear signal would include viscous normal traction.',
        'runtime_seconds':time.monotonic()-start,
        'source_sha256':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TASK.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}}
    (BASE/'results/acoustic-cavity-pressure-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
