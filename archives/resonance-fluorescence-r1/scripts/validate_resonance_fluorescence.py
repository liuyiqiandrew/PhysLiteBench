"""Validate fluorescence count variance and independent detector counting models."""
from pathlib import Path
import argparse,hashlib,importlib.util,itertools,json,time
import numpy as np

BASE=Path(__file__).resolve().parents[1]
TASK=BASE/'tasks/resonance-fluorescence'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def normalized(actual,truth):
    return float(np.linalg.norm(actual-truth)/np.linalg.norm(truth))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    oracle=load(TASK/'solution/model.py','oracle');source=load(BASE/'scripts/resonance_fluorescence_baseline.py','source')
    ref=load(TASK/'tests/reference.py','reference');proto=load(BASE/'prototypes/prototype.py','prototype')
    meta=json.loads((TASK/'tests/metadata.json').read_text());sigma=meta['sigma'];true=ref.TRUE_PARAMETER
    inputs=ref.calibration_inputs();clean=ref.predict(inputs)
    if args.generate:
        rng=np.random.default_rng(meta['calibration_seed'])
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,clean+rng.normal(0,sigma,len(inputs)))]
        text=json.dumps(records,indent=2)+'\n'
        for p in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:p.write_text(text)
    records=json.loads((TASK/'tests/data/calibration.json').read_text())
    assert records==json.loads((TASK/'environment/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['input']['observable']=='mean' for r in records)
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==.0004 for r in records)
    assert meta['prediction_limit']==.04
    hidden=ref.hidden_inputs();truth={k:ref.predict(es) for k,es in hidden.items()};variance=[k for k in hidden if k!='mean_anchors']
    def evaluate(module,rr):
        model=module.Model().fit(rr);values=np.array([r['value'] for r in rr])
        return dict(efficiency=model.efficiency,parameter_relative_error=abs(model.efficiency/true-1),
                    calibration_chi2=float(np.sum(((model.predict(inputs)-values)/sigma)**2)/(len(inputs)-1)),
                    hidden={k:normalized(model.predict(es),truth[k]) for k,es in hidden.items()})
    nominal={name:evaluate(module,records) for name,module in [('oracle',oracle),('shortcut',source)]}
    noise=[];rng=np.random.default_rng(meta['noise_validation_seed'])
    for _ in range(256):
        rr=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,clean+rng.normal(0,sigma,len(inputs)))]
        results={name:evaluate(module,rr) for name,module in [('oracle',oracle),('shortcut',source)]}
        for result in results.values():assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5
        assert max(results['oracle']['hidden'].values())<.04
        assert min(results['shortcut']['hidden'][k] for k in variance)>.04
        noise.append(results)
    # The calibration is a nonzero known basis times one positive efficiency.
    recover=[]
    for eta in [.5,.6,.73,.8,.9]:
        yy=ref.predict(inputs,eta);rr=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,yy)]
        recover.append(abs(oracle.Model().fit(rr).efficiency/eta-1))
    assert max(recover)<1e-14
    eq=float(np.max(abs(oracle.predict_at(inputs,true)-source.predict_at(inputs,true))))
    assert eq==0
    cases=list(itertools.product([.6,1.8],[-.5,0.,.5],[.8,1.2],[1.,6.],[.5,.9]))
    rng=np.random.default_rng(241347)
    cases += [(float(rng.uniform(.6,1.8)),float(rng.uniform(-.5,.5)),float(rng.uniform(.8,1.2)),float(rng.uniform(1,6)),float(rng.uniform(.5,.9))) for _ in range(16)]
    errors=[];contour=[];refinement=[];covariance=[];stationarity=[];trace=[];thinning=[];symmetries=[];signals=[];excess=[]
    for r,d,g,t,eta in cases:
        exact=np.array(oracle.response(r,d,g,t,eta));reference=np.array(ref.count_moments(r,d,g,t,eta));approx=np.array(source.response(r,d,g,t,eta))
        tilted=np.array(proto.tilted_contour(r,d,g,t,eta));tilted2=np.array(proto.tilted_contour(r,d,g,t,eta,points=48,radius=.06))
        errors.append(float(np.max(abs(exact-reference))));contour.append(float(np.max(abs(exact-tilted))));refinement.append(float(np.max(abs(tilted2-tilted))))
        l,rho=oracle.generator(r,d,g);stationarity.append(float(np.max(abs(l@rho.reshape(4,order='F')))))
        trace.append(float(abs(np.trace(rho)-1)));covariance.append(float(np.linalg.eigvalsh(rho).min()))
        m1,v1=oracle.response(r,d,g,t,1.);thinning.append(abs(exact[1]-(eta*m1+eta*eta*(v1-m1))))
        symmetries.append(float(np.max(abs(exact-np.array(oracle.response(r,-d,g,t,eta))))))
        signals.append(float(exact[1]));excess.append(float(approx[1]-approx[0]))
    assert max(errors)<1e-9 and max(contour)<1e-9 and max(refinement)<1e-9
    assert min(signals)>0 and min(excess)>=0 and min(covariance)>0
    assert max(stationarity)<1e-12 and max(trace)<1e-12 and max(thinning)<1e-12 and max(symmetries)<1e-12
    resolved=[]
    for case in [(1.,0.,1.,6.,.73),(.6,.5,.8,3.,.5),(1.8,-.5,1.2,6.,.9)]:
        exact=np.array(oracle.response(*case));m,v,checks=proto.resolved_counts(*case,cutoff=48)
        mr,vr,cr=proto.resolved_counts(*case,cutoff=64)
        resolved.append(dict(case=case,error=float(np.max(abs(exact-[m,v]))),cutoff_change=float(max(abs(m-mr),abs(v-vr))),**checks))
    assert max(x['error'] for x in resolved)<1e-9 and max(x['cutoff_change'] for x in resolved)<1e-12
    dark=[];short=[]
    for d,g,eta in itertools.product([-.5,.5],[.8,1.2],[.5,.9]):
        dark.append(float(np.max(np.abs(oracle.response(0.,d,g,3.,eta)))))
        m,v=oracle.response(1.,d,g,1e-6,eta);short.append(abs(v/m-1))
        assert np.max(np.abs(oracle.response(1.,d,g,3.,0.)))==0
    assert max(dark)<1e-14 and max(short)<1e-6
    report=dict(task='resonance-fluorescence',revision=1,model_evaluations=0,noise_trials=256,controls=nominal,
        noise=dict(oracle_passes=256,shortcut_rejections=256,all_calibration_and_parameter_pass=True,
           max_parameter_relative_error=max(x[k]['parameter_relative_error'] for x in noise for k in x),
           max_calibration_chi2=max(x[k]['calibration_chi2'] for x in noise for k in x),
           max_oracle_hidden_error=max(v for x in noise for v in x['oracle']['hidden'].values()),
           min_shortcut_variance_error=min(x['shortcut']['hidden'][k] for x in noise for k in variance)),
        physics=dict(independent_domain_cases=len(cases),direct_count_moment_error_max=max(errors),
           independent_contour_error_max=max(contour),contour_refinement_error_max=max(refinement),
           stationary_density_eigenvalue_min=min(covariance),stationarity_residual_max=max(stationarity),trace_error_max=max(trace),
           thinning_identity_error_max=max(thinning),detuning_reversal_error_max=max(symmetries),
           count_variance_range=[min(signals),max(signals)],source_excess_variance_min=min(excess),
           number_resolved_checks=resolved,dark_limit_error_max=max(dark),short_gate_poisson_limit_error_max=max(short),
           mean_calibration_equivalence_max=eq,minimum_hidden_variance=min(float(np.min(truth[k])) for k in variance)),
        identifiability=dict(basis='A known positive mean-count basis times one efficiency; weighted least squares has a strictly positive constant Hessian.',
           minimum_calibration_basis=float(np.min(clean/true)),weighted_design_norm=float(np.dot(clean/true,clean/true)/sigma**2),max_noiseless_parameter_error=max(recover)),
        data_integrity=dict(records=len(records),unique_settings=len({json.dumps(e,sort_keys=True) for e in inputs}),fixed_instrument_sigma=sigma,
           public_private_identical=True,sigma_independent_of_response_and_parameter=True,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_validation_seed'],domain_seed=241347),
        runtime_seconds=time.monotonic()-start,
        source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TASK.rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
    (BASE/'results/resonance-fluorescence-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
