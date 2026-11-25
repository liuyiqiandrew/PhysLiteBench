"""Calibrate and validate polymer stress against direct molecular force integrals."""
from pathlib import Path
import argparse,hashlib,importlib.util,itertools,json,time
import numpy as np
from scipy.special import roots_jacobi,iv

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
    shortcut=load(BASE/'scripts/fene_stress_baseline.py','shortcut')
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
    nominal={name:evaluate(module,records) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    noise=[];rng=np.random.default_rng(meta['noise_validation_seed'])
    for _ in range(256):
        rr=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,clean+rng.normal(0,sigma,len(inputs)))]
        results={name:evaluate(module,rr) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
        for result in results.values():assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5
        assert max(results['oracle']['hidden'].values())<.04
        assert min(results['shortcut']['hidden'][k] for k in finite)>.04
        noise.append(results)
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
        for e in inputs[:18]:
            r=z*e['rate']/2
            sensitivities.append(abs(e['temperature']*e['rate']*(1+r*r)/(1-r*r)**2))
    equivalence=float(np.max(abs(oracle.predict_at(inputs,true)-shortcut.predict_at(inputs,true))))
    assert equivalence<1e-14 and min(sensitivities)>0
    # Public-domain corners, near-zero flow, both signs and interior points.
    cases=list(itertools.product([-.95,-.04,.04,.95],[.8,1.2],[3.2,4.8],[np.sqrt(6),np.sqrt(12)]))
    rng=np.random.default_rng(241147)
    cases += [(float(rng.uniform(-.95,.95)),float(rng.uniform(.8,1.2)),float(rng.uniform(3.2,4.8)),float(rng.uniform(np.sqrt(6),np.sqrt(12)))) for _ in range(24)]
    comparisons=[];source_residual=[];extensions=[];force_error=[];currents=[];reverse=[];dissipation=[]
    for rate,temp,z,length in cases:
        exp=ref.experiment(rate,temp,length)
        exact,c=oracle.stress_and_tensor(rate,temp,z,length)
        source_c,f=shortcut.conformation(rate,temp,z,length)
        independent,cr,current=ref.cartesian(rate,temp,z,length,80)
        refined,_,_=ref.cartesian(rate,temp,z,length,128)
        comparisons.append(dict(reference_relative_error=abs(independent/exact-1),refinement_relative_change=abs(refined/independent-1)))
        currents.append(current)
        kappa=np.diag([rate,-rate]);residual=kappa@source_c+source_c@kappa-4/z*(f*source_c-temp*np.eye(2))
        source_residual.append(float(np.max(abs(residual))))
        extensions += [float(np.trace(source_c)/length**2),float(np.trace(c)/length**2)]
        # A separate weighted radial integral computes the singular force itself.
        b=length**2/temp;wi=z*rate/4
        x,w=roots_jacobi(96,b/2,0);u=(x+1)/2;den=np.dot(w,iv(0,wi*b*u))
        x,w=roots_jacobi(96,b/2-1,0);u=(x+1)/2
        force=2*temp*b*np.dot(w,u*iv(1,wi*b*u))/den
        force_error.append(abs(force/exact-1))
        for module in [oracle,shortcut]:
            es=[exp,dict(exp,rate=-rate)]
            yy=module.predict_at(es,z)
            reverse.append(abs(yy.sum()))
            dissipation.append(float(rate*yy[0]))
    assert max(x['reference_relative_error'] for x in comparisons)<2e-6
    assert max(x['refinement_relative_change'] for x in comparisons)<2e-6
    assert max(force_error)<1e-10 and max(source_residual)<1e-10
    assert min(extensions)>0 and max(extensions)<1
    assert max(reverse)<1e-10 and min(dissipation)>=0 and max(currents)<1e-7
    equilibrium=[]
    for temp,z,length in itertools.product([.8,1.2],[3.2,4.8],[np.sqrt(6),np.sqrt(12)]):
        stress,c=oracle.stress_and_tensor(0.,temp,z,length)
        value,cr,_=ref.cartesian(0.,temp,z,length,128)
        b=length**2/temp
        equilibrium.append(float(max(abs(stress),abs(value),abs(np.trace(c)-2*temp*b/(b+4)),np.max(abs(c-cr)))))
    assert max(equilibrium)<1e-8
    report=dict(task='fene-stress',revision=1,model_evaluations=0,noise_trials=256,controls=nominal,
        noise=dict(oracle_passes=256,shortcut_rejections=256,all_calibration_and_parameter_pass=True,
           max_parameter_relative_error=max(x[k]['parameter_relative_error'] for x in noise for k in x),
           max_calibration_chi2=max(x[k]['calibration_chi2'] for x in noise for k in x),
           max_oracle_hidden_error=max(v for x in noise for v in x['oracle']['hidden'].values()),
           min_shortcut_finite_error=min(x['shortcut']['hidden'][k] for x in noise for k in finite)),
        physics=dict(independent_domain_cases=len(cases),reference_relative_error_max=max(x['reference_relative_error'] for x in comparisons),
           reference_refinement_relative_error_max=max(x['refinement_relative_change'] for x in comparisons),
           independent_force_average_relative_error_max=max(force_error),source_constitutive_residual_max=max(source_residual),
           extension_fraction_range=[min(extensions),max(extensions)],stationary_current_max=max(currents),
           reversal_error_max=max(reverse),minimum_flow_power=min(dissipation),equilibrium_error_max=max(equilibrium),
           hookean_calibration_equivalence_max=equivalence,minimum_hookean_relaxation_margin=2/4.8-.25,
           finite_extensibility_b_range=[5,15],force_moment_boundary_exponents={'first_min':1.5,'second_min':.5},
           minimum_hidden_signal=min(float(np.min(abs(truth[k]))) for k in finite),maximum_hidden_signal=max(float(np.max(abs(truth[k]))) for k in finite)),
        identifiability=dict(basis='Each nonzero Hookean response has a derivative of fixed sign in positive drag; full-range objective profiles and noiseless endpoint recovery checked.',
           minimum_nonzero_drag_derivative=min(sensitivities),max_noiseless_parameter_error=max(recover),objective_profiles=profiles),
        data_integrity=dict(records=len(records),unique_settings=len({json.dumps(e,sort_keys=True) for e in inputs}),fixed_instrument_sigma=sigma,
           public_private_identical=True,sigma_independent_of_response_and_parameter=True,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_validation_seed'],domain_seed=241147),
        runtime_seconds=time.monotonic()-start,
        source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TASK.rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
    (BASE/'results/fene-stress-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
