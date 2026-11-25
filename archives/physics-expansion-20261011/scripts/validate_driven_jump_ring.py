"""Validate driven jump ring r1; --generate intentionally replaces simulated data."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/driven-jump-ring'


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual,expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def homogeneous(e,kappa):
    value=0.
    for s in e['stages']:
        factor=kappa*np.exp(s['energies'][0]-s['barriers'][0])
        value+=factor*s['duration']*(np.exp(s['affinity']/6)*np.expm1(e['field']/3)
                                   +np.exp(-s['affinity']/6)*np.expm1(-e['field']/3))
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    started=time.monotonic()
    ref=load('ref',TASK/'tests/reference.py')
    oracle=load('oracle',TASK/'solution/model.py')
    baseline=load('baseline',ROOT/'scripts/driven_jump_ring_baseline.py')
    kappa=ref.TRUE_PARAMETER
    inputs=ref.calibration_inputs()*5
    means=np.array([homogeneous(e,kappa) for e in inputs])
    assert np.max(abs(ref.predict(inputs,kappa)-means)) < 1e-12
    if args.generate:
        rng=np.random.default_rng(611103)
        records=[dict(input=e,value=float(y),sigma=.00015)
                 for e,y in zip(inputs,means+rng.normal(0,.00015,len(means)))]
        for part in ['environment','tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
        truth={name:dict(inputs=rows,values=ref.predict(rows,kappa).tolist())
               for name,rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    truth=json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    for p in np.linspace(.8,1.3,11):
        analytical=np.array([homogeneous(e,p) for e in inputs])
        assert np.max(abs(oracle.predict_at(inputs,p)-analytical)) < 1e-12
        assert np.max(abs(baseline.predict_at(inputs,p)-analytical)) < 1e-12
    models={name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',baseline)]}
    report={'revision':1,'calibration_seed':611103,'noise_seed':611107,'controls':{},'groups':{}}
    for name,model in models.items():
        residual=(model.predict(inputs)-[r['value'] for r in records])/.00015
        report['controls'][name]=dict(attempt_rate=model.attempt_rate,
                                     chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2'] < 1.5
        assert abs(model.attempt_rate/kappa-1)<.03
    rng=np.random.default_rng(611107)
    basis=means/kappa
    fits=[]; chis=[]
    # This is the exact fitting operation of both models on this calibration;
    # explicit Model.fit checks above also exercise the implementations.
    for _ in range(256):
        values=means+rng.normal(0,.00015,len(means))
        fitted=float(basis@values/(basis@basis))
        fits.append(fitted)
        residual=(basis*fitted-values)/.00015
        chis.append(float(residual@residual)/(len(values)-1))
    assert max(abs(np.array(fits)/kappa-1)) < .03
    report.update(noise_draws=256,noise_parameter_range=[min(fits),max(fits)],noise_max_chi2=max(chis))
    for name,rows in ref.hidden_inputs().items():
        expected=np.array(truth[name]['values'])
        fresh=ref.predict(rows,kappa)
        refined=ref.predict(rows,kappa,tolerance=5e-16,series_tolerance=2e-18,initial=[1.,0.,0.])
        assert np.max(abs(expected-fresh)) < 1e-12
        assert np.max(abs(fresh-refined)) < 1e-10
        assert np.max(abs(oracle.predict_at(rows,kappa)-refined)) < 1e-10
        oracle_noise=max(error(oracle.predict_at(rows,p),expected) for p in fits)
        shortcut_noise=min(error(baseline.predict_at(rows,p),expected) for p in fits)
        assert oracle_noise < .04
        if name!='uniform_anchors':
            assert shortcut_noise > .04
        report['groups'][name]=dict(errors={n:error(m.predict(rows),expected) for n,m in models.items()},
                                   noise_oracle_max=oracle_noise,noise_shortcut_min=shortcut_noise,
                                   min_abs_signal=float(min(abs(expected))),
                                   refinement_max=float(max(abs(fresh-refined))))
    rng=np.random.default_rng(611109)
    cases=[]
    for i in range(24):
        stages=[dict(energies=rng.uniform(0,1.8,3).tolist(),barriers=rng.uniform(2,4,3).tolist(),
                     affinity=float(rng.uniform(-2,2)),duration=float(rng.uniform(.1,1.5)))
                for j in range(1+i%4)]
        cases.append(dict(stages=stages,field=float(rng.uniform(-2,2))))
    differences=[]
    for p in [.8,1.1,1.3]:
        differences.extend(abs(oracle.predict_at(cases,p)-ref.predict(cases,p)))
    assert max(differences)<1e-10
    report['domain_cases']=72
    report['domain_max_absolute_difference']=float(max(differences))
    for e in cases:
        assert abs(oracle.response(dict(e,field=0.),kappa))<1e-12
        original=oracle.response(e,kappa)
        for shift in range(len(e['stages'])):
            shifted=dict(e,stages=e['stages'][shift:]+e['stages'][:shift])
            assert abs(oracle.response(shifted,kappa)-original)<1e-12
        one=dict(e,stages=e['stages'][:1])
        reversed_field=dict(one,field=-one['stages'][0]['affinity']-one['field'])
        assert abs(oracle.response(one,kappa)-oracle.response(reversed_field,kappa))<1e-12
        stage=dict(e['stages'][0],affinity=0.)
        p=np.exp(-np.asarray(stage['energies']));p/=p.sum()
        assert np.max(abs(oracle.generator(stage,kappa,0.)@p))<1e-12
    report['invariants']=['zero field','cycle origin','static fluctuation symmetry','equilibrium detailed balance']
    report['wall_seconds']=time.monotonic()-started
    (ROOT/'results/expansion-driven-jump-ring-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
