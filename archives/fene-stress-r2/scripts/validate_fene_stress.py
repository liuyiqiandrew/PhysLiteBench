"""Validate rotating-flow polymer stress and independent stationary probability balance."""
from pathlib import Path
import argparse,hashlib,importlib.util,itertools,json,time
import numpy as np

BASE=Path(__file__).resolve().parents[1]
TASK=BASE/'tasks/fene-stress'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def normalized(actual,truth):
    return float(np.linalg.norm(actual-truth)/np.linalg.norm(truth))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    oracle=load(TASK/'solution/model.py','oracle')
    source=load(BASE/'scripts/fene_stress_baseline.py','source')
    ref=load(TASK/'tests/reference.py','reference')
    meta=json.loads((TASK/'tests/metadata.json').read_text());sigma=meta['sigma'];true=ref.TRUE_PARAMETER
    inputs=ref.calibration_inputs();clean=ref.predict(inputs)
    if args.generate:
        rng=np.random.default_rng(meta['calibration_seed'])
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,clean+rng.normal(0,sigma,len(inputs)))]
        text=json.dumps(records,indent=2)+'\n'
        for p in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:p.write_text(text)
    records=json.loads((TASK/'tests/data/calibration.json').read_text())
    assert records==json.loads((TASK/'environment/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==.0005 for r in records)
    assert meta['prediction_limit']==.04
    hidden=ref.hidden_inputs();truth={k:ref.predict(es) for k,es in hidden.items()};finite=[k for k in hidden if k!='hookean_anchors']
    def evaluate(module,rr):
        model=module.Model().fit(rr);values=np.array([r['value'] for r in rr])
        return dict(drag=model.drag,parameter_relative_error=abs(model.drag/true-1),
                    calibration_chi2=float(np.sum(((model.predict(inputs)-values)/sigma)**2)/(len(inputs)-1)),
                    hidden={k:normalized(model.predict(es),truth[k]) for k,es in hidden.items()})
    nominal={name:evaluate(module,records) for name,module in [('oracle',oracle),('shortcut',source)]}
    print('nominal',nominal,flush=True)
    noise=[];rng=np.random.default_rng(meta['noise_validation_seed'])
    for trial in range(256):
        rr=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,clean+rng.normal(0,sigma,len(inputs)))]
        results={name:evaluate(module,rr) for name,module in [('oracle',oracle),('shortcut',source)]}
        for result in results.values():assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5
        assert max(results['oracle']['hidden'].values())<.04
        assert min(results['shortcut']['hidden'][k] for k in finite)>.04
        noise.append(results)
        if trial%64==0:print('noise',trial,flush=True)
    profiles=[];recover=[]
    for drag in [3.2,3.6,4.1,4.4,4.8]:
        yy=oracle.predict_at(inputs,drag)
        rr=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,yy)]
        recover.append(abs(oracle.Model().fit(rr).drag/drag-1))
        grid=np.linspace(3.2,4.8,321);loss=np.array([np.sum((oracle.predict_at(inputs,z)-yy)**2) for z in grid])
        profiles.append(dict(truth=drag,grid_minimum=float(grid[np.argmin(loss)])))
        assert abs(grid[np.argmin(loss)]-drag)<.002501
    assert max(recover)<2e-8
    sensitivities=[]
    for z in np.linspace(3.2,4.8,33):
        for e in inputs[:24]:
            k=(e['rotation']**2-e['rate']**2)/4
            sensitivities.append(abs(e['temperature']*e['rate']*(1-k*z*z)/(1+k*z*z)**2))
    equivalence=float(np.max(abs(oracle.predict_at(inputs,true)-source.predict_at(inputs,true))))
    assert equivalence<1e-14 and min(sensitivities)>0
    corners=list(itertools.product([-.95,.95],[-1.2,1.2],[.8,1.2],[3.2,4.8],[np.sqrt(6),np.sqrt(12)]))
    rng=np.random.default_rng(241247)
    corners += [(float(rng.uniform(-.95,.95)),float(rng.uniform(-1.2,1.2)),float(rng.uniform(.8,1.2)),float(rng.uniform(3.2,4.8)),float(rng.uniform(np.sqrt(6),np.sqrt(12)))) for _ in range(16)]
    comparisons=[];residuals=[];extensions=[];reversals=[];powers=[];tail=[];mass=[];positive=[];conservation=[]
    for i,(rate,rotation,temp,z,length) in enumerate(corners):
        exact,c,d=oracle.spectral(rate,rotation,temp,z,length)
        refined,_,_=oracle.spectral(rate,rotation,temp,z,length,degree=28)
        coarse,_,_=ref.finite_volume(rate,rotation,temp,z,length,nr=64,nt=96)
        fine,_,fd=ref.finite_volume(rate,rotation,temp,z,length,nr=128,nt=192)
        independent=(4*fine-coarse)/3
        comparisons.append(dict(reference_relative_error=abs(independent/exact-1),
                                reference_refinement_relative_change=abs((fine-coarse)/exact),
                                spectral_refinement_relative_change=abs((refined-exact)/exact)))
        sc,f=source.conformation(rate,rotation,temp,z,length);kappa=np.array([[rate,-rotation],[rotation,-rate]])
        residuals.append(float(np.max(abs(kappa@sc+sc@kappa.T-4/z*(f*sc-temp*np.eye(2))))))
        extensions += [float(np.trace(sc)/length**2),float(np.trace(c)/length**2)]
        tail.append(d['negative_probability_mass']);mass.append(d['normalization_error'])
        positive.append(fd['minimum_probability']);conservation.append(fd['conservation'])
        exp=ref.experiment(rate,rotation,temp,length)
        for module in [oracle,source]:
            yy=module.predict_at([exp,dict(exp,rate=-rate),dict(exp,rotation=-rotation)],z)
            reversals += [abs(yy[0]+yy[1]),abs(yy[0]-yy[2])]
            powers.append(float(rate*yy[0]))
        if i%16==0:print('domain',i,flush=True)
    assert max(x['reference_relative_error'] for x in comparisons)<3e-4
    assert max(x['spectral_refinement_relative_change'] for x in comparisons)<1e-8
    assert max(residuals)<1e-10 and 0<min(extensions)<max(extensions)<1
    assert max(reversals)<1e-9 and min(powers)>0
    assert min(positive)>0 and max(conservation)<1e-9
    assert max(tail)<1e-5 and max(mass)<1e-10
    limits=[]
    for temp,z,length,rotation in itertools.product([.8,1.2],[3.2,4.8],[np.sqrt(6),np.sqrt(12)],[0.,.8]):
        stress,c,_=oracle.spectral(0.,rotation,temp,z,length)
        expected=temp/(1+4*temp/length**2)*np.eye(2)
        limits.append(float(max(abs(stress),np.max(abs(c-expected)))))
    assert max(limits)<1e-9
    # Nonzero circulation is measured in the independent stationary current balance.
    _,_,circulation=ref.finite_volume(.8,.8,1.,true,3.,nr=80,nt=128)
    assert circulation['current_l1']>1
    report=dict(task='fene-stress',revision=2,model_evaluations=0,noise_trials=256,controls=nominal,
        noise=dict(oracle_passes=256,shortcut_rejections=256,all_calibration_and_parameter_pass=True,
           max_parameter_relative_error=max(x[k]['parameter_relative_error'] for x in noise for k in x),
           max_calibration_chi2=max(x[k]['calibration_chi2'] for x in noise for k in x),
           max_oracle_hidden_error=max(v for x in noise for v in x['oracle']['hidden'].values()),
           min_shortcut_finite_error=min(x['shortcut']['hidden'][k] for x in noise for k in finite)),
        physics=dict(independent_domain_cases=len(corners),reference_relative_error_max=max(x['reference_relative_error'] for x in comparisons),
           reference_refinement_relative_error_max=max(x['reference_refinement_relative_change'] for x in comparisons),
           spectral_refinement_relative_error_max=max(x['spectral_refinement_relative_change'] for x in comparisons),
           source_constitutive_residual_max=max(residuals),extension_fraction_range=[min(extensions),max(extensions)],
           reversal_error_max=max(reversals),minimum_flow_power=min(powers),equilibrium_and_pure_rotation_error_max=max(limits),
           hookean_calibration_equivalence_max=equivalence,minimum_hookean_relaxation_margin=2/4.8-.25,
           finite_extensibility_b_range=[5,15],force_moment_boundary_exponents={'first_min':1.5,'second_min':.5},
           spectral_negative_tail_mass_max=max(tail),spectral_normalization_error_max=max(mass),
           finite_volume_probability_min=min(positive),finite_volume_conservation_error_max=max(conservation),
           mixed_flow_stationary_current_l1=circulation['current_l1'],
           minimum_hidden_signal=min(float(np.min(abs(truth[k]))) for k in finite),maximum_hidden_signal=max(float(np.max(abs(truth[k]))) for k in finite)),
        identifiability=dict(basis='Every chosen Hookean calibration response has a nonzero fixed-sign drag derivative across the fitted interval; full-range objective profiles and endpoint recovery checked.',
           minimum_nonzero_drag_derivative=min(sensitivities),max_noiseless_parameter_error=max(recover),objective_profiles=profiles),
        data_integrity=dict(records=len(records),unique_settings=len({json.dumps(e,sort_keys=True) for e in inputs}),fixed_instrument_sigma=sigma,
           public_private_identical=True,sigma_independent_of_response_and_parameter=True,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_validation_seed'],domain_seed=241247),
        runtime_seconds=time.monotonic()-start,
        source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TASK.rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
    (BASE/'results/fene-stress-r2-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
