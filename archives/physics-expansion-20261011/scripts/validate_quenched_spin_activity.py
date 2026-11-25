"""Validate quenched spin activity r1; --generate intentionally replaces simulated data."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/quenched-spin-activity'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual,expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def uncoupled(e,kappa):
    fields=e['field']+np.array([e['disorder'],-e['disorder']])
    biases=np.array([e['bias_plus'],e['bias_minus']])
    up=.5*(1+np.tanh(fields));down=1-up
    value=0.
    for a,b,s in zip(up,down,biases):
        matrix=np.array([[-a,b*np.exp(-s)],[a*np.exp(-s),-b]])
        value+=np.max(np.linalg.eigvals(matrix).real)/2
    return kappa*value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    started=time.monotonic()
    ref=load('ref',TASK/'tests/reference.py')
    oracle=load('oracle',TASK/'solution/model.py')
    baseline=load('baseline',ROOT/'scripts/quenched_spin_activity_baseline.py')
    rate=ref.TRUE_PARAMETER
    calibration=ref.calibration_inputs()*8
    means=np.array([uncoupled(e,rate) for e in calibration])
    assert np.max(abs(oracle.predict_at(calibration,rate)-means))<1e-12
    assert np.max(abs(baseline.predict_at(calibration,rate)-means))<1e-12
    if args.generate:
        rng=np.random.default_rng(611201)
        records=[dict(input=e,value=float(y),sigma=.0003)
                 for e,y in zip(calibration,means+rng.normal(0,.0003,len(means)))]
        for part in ['environment','tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
        truth={name:dict(inputs=rows,values=ref.predict(rows,rate).tolist())
               for name,rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    truth=json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models={name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',baseline)]}
    report={'revision':1,'calibration_seed':611201,'noise_seed':611203,'controls':{},'groups':{}}
    for name,model in models.items():
        residual=(model.predict(calibration)-[r['value'] for r in records])/.0003
        report['controls'][name]=dict(attempt_rate=model.attempt_rate,chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2']<1.5
        assert abs(model.attempt_rate/rate-1)<.03
    rng=np.random.default_rng(611203);fits=[];chis=[]
    basis=means/rate
    for _ in range(256):
        values=means+rng.normal(0,.0003,len(means))
        fitted=float(basis@values/(basis@basis));fits.append(fitted)
        residual=(basis*fitted-values)/.0003
        chis.append(float(residual@residual)/(len(values)-1))
    report.update(noise_draws=256,noise_parameter_range=[min(fits),max(fits)],noise_max_chi2=max(chis))
    assert max(abs(np.asarray(fits)/rate-1))<.03
    for name,rows in ref.hidden_inputs().items():
        expected=np.asarray(truth[name]['values'])
        fresh=ref.predict(rows,rate)
        refined=ref.predict(rows,rate,base_size=160)
        physical=oracle.predict_at(rows,rate)
        grid_refined=np.array([oracle.response(e,rate,grid=97) for e in rows])
        assert np.max(abs(fresh-expected))<1e-9
        assert error(fresh,refined)<.001
        assert error(physical,refined)<.001
        assert np.max(abs(physical-grid_refined))<1e-8
        shortcut=baseline.predict_at(rows,rate)
        errors={key:error((physical if key=='oracle' else shortcut)*m.attempt_rate/rate,expected)
                for key,m in models.items()}
        noise_oracle=max(error(physical*p/rate,expected) for p in fits)
        noise_shortcut=min(error(shortcut*p/rate,expected) for p in fits)
        assert noise_oracle<.04
        if name!='uncoupled_anchors':assert noise_shortcut>.04
        report['groups'][name]=dict(errors=errors,noise_oracle_max=noise_oracle,noise_shortcut_min=noise_shortcut,
                                   reference_refinement_error=error(fresh,refined),oracle_reference_error=error(physical,refined),
                                   grid_refinement_max=float(max(abs(physical-grid_refined))),
                                   min_abs_signal=float(min(abs(expected))))
    rng=np.random.default_rng(611207)
    cases=[ref.experiment(coupling=float(rng.uniform(0,1.6)),field=float(rng.uniform(-.5,.5)),
                          disorder=float(rng.uniform(0,1.2)),bias_plus=float(rng.uniform(-1,1.5)),
                          bias_minus=float(rng.uniform(-1,1.5))) for _ in range(24)]
    cases.extend([ref.experiment(coupling=j,field=0.,disorder=d,bias_plus=s,bias_minus=s)
                  for j in [0.,1.,1.6] for d in [0.,1.2] for s in [-1.,1.5]])
    physical=oracle.predict_at(cases,1.)
    refined=np.array([oracle.response(e,1.,grid=97) for e in cases])
    finite=ref.predict(cases,1.,base_size=160)
    assert np.max(abs(physical-refined))<1e-8
    assert np.max(abs(physical-finite))<1e-4
    for e,value in zip(cases,physical):
        reflected=dict(e,field=-e['field'],bias_plus=e['bias_minus'],bias_minus=e['bias_plus'])
        assert abs(oracle.response(reflected,1.)-value)<1e-8
        assert abs(oracle.response(dict(e,bias_plus=0.,bias_minus=0.),1.))<1e-8
    report.update(domain_cases=len(cases),domain_reference_max_absolute_difference=float(max(abs(physical-finite))),
                  domain_grid_refinement_max=float(max(abs(physical-refined))),invariants=['zero biases','spin reflection and population exchange'],
                  wall_seconds=time.monotonic()-started)
    (ROOT/'results/expansion-quenched-spin-activity-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
