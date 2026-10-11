"""Validate the counterion cell; --generate intentionally replaces data."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/counterion-cloud'


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual,expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    args = parser.parse_args()
    ref = load('reference',TASK/'tests/reference.py')
    oracle = load('oracle',TASK/'solution/model.py')
    shortcut = load('shortcut',ROOT/'scripts/counterion_cloud_baseline.py')
    density,sigma = ref.TRUE_PARAMETER,.001
    inputs = ref.calibration_inputs()*10
    means = ref.predict(inputs,density)
    assert max(abs(oracle.predict_at(inputs,density)-means)) < 1e-12
    assert max(abs(shortcut.predict_at(inputs,density)-means)) < 1e-12
    if args.generate:
        rng = np.random.default_rng(618001)
        records = [dict(input=e,value=float(y),sigma=sigma)
                   for e,y in zip(inputs,means+rng.normal(0,sigma,len(means)))]
        for part in ['environment','tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
        truth = {name:dict(inputs=rows,values=ref.predict(rows,density).tolist())
                 for name,rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    assert inputs == [r['input'] for r in records]
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    report = dict(revision=1,calibration_seed=618001,noise_seed=618009,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        chi = float(residual@residual)/(len(records)-1)
        report['controls'][name] = dict(line_density=model.line_density,chi2=chi)
        assert chi < 1.5 and abs(model.line_density/density-1) < .03
    rng = np.random.default_rng(618009)
    fits,chis = [],[]
    for _ in range(256):
        noisy = [dict(r,value=float(y)) for r,y in zip(records,means+rng.normal(0,sigma,len(means)))]
        model = oracle.Model().fit(noisy)
        fits.append(model.line_density)
        residual = (model.predict(inputs)-[r['value'] for r in noisy])/sigma
        chis.append(float(residual@residual)/(len(noisy)-1))
    assert max(chis) < 1.5 and max(abs(np.asarray(fits)/density-1)) < .03
    report.update(noise_draws=256,noise_parameter_range=[min(fits),max(fits)],noise_max_chi2=max(chis))
    for name,rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        assert rows == truth[name]['inputs']
        fresh = ref.predict(rows,density)
        refined = ref.predict(rows,density,length=1600.)
        assert max(abs(fresh-expected)) < 1e-12
        assert max(abs(fresh-refined)) < 1e-10
        assert max(abs(oracle.predict_at(rows,density)-refined)) < 1e-10
        errors = {key:error(m.predict(rows),expected) for key,m in models.items()}
        noise_oracle = max(error(oracle.predict_at(rows,p),expected) for p in fits)
        noise_shortcut = min(error(shortcut.predict_at(rows,p),expected) for p in fits)
        assert errors['oracle'] < .04 and noise_oracle < .04
        if name != 'shared_anchors':
            assert errors['shortcut'] > .04 and noise_shortcut > .04
        report['groups'][name] = dict(errors=errors,noise_oracle_max=noise_oracle,
                                     noise_shortcut_min=noise_shortcut,
                                     reference_refinement_max_error=float(max(abs(fresh-refined))))
    recovery = 0.
    for tau in np.linspace(.7,1.4,41):
        noiseless = [dict(r,value=float(y)) for r,y in zip(records,oracle.predict_at(inputs,tau))]
        recovery = max(recovery,abs(oracle.Model().fit(noiseless).line_density-tau))
    assert recovery < 1e-12
    rng = np.random.default_rng(618013)
    cases = [(ref.experiment(float(rng.uniform(.2,1)),float(rng.uniform(.6,1.25)),
                            float(rng.uniform(.5,2))),float(rng.uniform(.7,1.4))) for _ in range(64)]
    cases += [(ref.experiment(u,b,a),tau) for u in [.2,1.] for b in [.6,1.25]
              for a in [.5,2.] for tau in [.7,1.4]]
    domain_error = max(abs(oracle.predict_at([e],tau)[0]-ref.predict([e],tau)[0]) for e,tau in cases)
    assert domain_error < 1e-6
    convergence = []
    for xi in [.084,.4,.8,.95,.999,1.,1.001,1.05,1.25,1.625,1.75]:
        e = ref.experiment(1.,xi)
        exact = oracle.predict_at([e],1.)[0]
        direct,charge = ref.bvp(xi,12.)
        finite,_ = ref.finite_cell(xi,12.)
        coarse,fine = ref.limiting(xi,800.),ref.limiting(xi,1600.)
        assert abs(direct-finite) < 1e-9 and abs(charge-1) < 1e-8
        assert abs(coarse-exact) < 1e-6 and abs(fine-exact) < 1e-6
        convergence.append(dict(xi=xi,bvp_first_integral_error=float(abs(direct-finite)),
                                neutral_charge_error=float(abs(charge-1)),
                                limit_error=float(abs(coarse-exact)),
                                refined_limit_error=float(abs(fine-exact))))
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_error=recovery,
                  domain_cases=len(cases),domain_reference_max_error=float(domain_error),
                  convergence=convergence,
                  reference='Finite-cell PB first integral at log(R/a)=800,1600,3200,6400; inverse-log extrapolation, factor-two refinement, and direct BVP checks',
                  invariants=['neutral cell','zero outer electric field','radius-independent dilution limit',
                              'ideal pressure as coupling vanishes','continuous value and slope at condensation'])
    path = ROOT/'results/expansion-counterion-cloud-r1-validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
