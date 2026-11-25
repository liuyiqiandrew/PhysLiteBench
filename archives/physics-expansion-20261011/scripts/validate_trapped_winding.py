"""Validate continuum winding rates; --generate replaces checked-in data."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.special import gammaln, hyp2f1

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/trapped-winding'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    ref = load('ref', TASK/'tests/reference.py')
    oracle = load('oracle', TASK/'solution/model.py')
    shortcut = load('shortcut', ROOT/'scripts/trapped_winding_baseline.py')
    mobility,sigma = ref.TRUE_PARAMETER,.001
    inputs = ref.calibration_inputs()*8
    means = ref.predict(inputs,mobility)
    assert max(abs(oracle.predict_at(inputs,mobility)-means)) < 1e-8
    assert max(abs(shortcut.predict_at(inputs,mobility)-means)) < 1e-8
    if args.generate:
        rng = np.random.default_rng(615001)
        records = [dict(input=e,value=float(y),sigma=sigma)
                   for e,y in zip(inputs,means+rng.normal(0,sigma,len(means)))]
        for part in ['environment','tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
        truth = {name:dict(inputs=rows,values=ref.predict(rows,mobility).tolist())
                 for name,rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    assert inputs == [r['input'] for r in records]
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    report = dict(revision=1,calibration_seed=615001,noise_seed=615009,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        chi = float(residual@residual)/(len(records)-1)
        report['controls'][name] = dict(mobility=model.mobility,chi2=chi)
        assert chi < 1.5 and abs(model.mobility/mobility-1) < .03
    rng = np.random.default_rng(615009)
    fits,chis = [],[]
    for _ in range(256):
        noisy = [dict(r,value=float(y)) for r,y in zip(records,means+rng.normal(0,sigma,len(means)))]
        model = oracle.Model().fit(noisy)
        fits.append(model.mobility)
        residual = (model.predict(inputs)-[r['value'] for r in noisy])/sigma
        chis.append(float(residual@residual)/(len(noisy)-1))
    assert max(chis) < 1.5 and max(abs(np.asarray(fits)/mobility-1)) < .03
    report.update(noise_draws=256,noise_parameter_range=[min(fits),max(fits)],noise_max_chi2=max(chis))
    for name,rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        assert rows == truth[name]['inputs']
        fresh = ref.predict(rows,mobility)
        refined = ref.predict(rows,mobility,times=(14.,18.),order=128)
        assert max(abs(fresh-expected)) < 1e-12
        assert max(abs(fresh-refined)) < 1e-8
        assert max(abs(oracle.predict_at(rows,mobility)-refined)) < 1e-10
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
    for mu in np.linspace(.7,1.4,41):
        noiseless = [dict(r,value=float(y)) for r,y in zip(records,oracle.predict_at(inputs,mu))]
        recovery = max(recovery,abs(oracle.Model().fit(noiseless).mobility-mu))
    assert recovery < 1e-12
    rng = np.random.default_rng(615013)
    cases = [(ref.experiment(float(rng.uniform(.6,1.8)),float(rng.uniform(.5,1.8)),
                            float(rng.uniform(-2.5,2.5))),float(rng.uniform(.7,1.4))) for _ in range(96)]
    cases += [(ref.experiment(k,t,p),mu) for k in [.6,1.8] for t in [.5,1.8]
              for p in [-2.5,-1.,0.,1.,2.5] for mu in [.7,1.4]]
    domain_error = 0.
    for e,mu in cases:
        value = oracle.predict_at([e],mu)[0]
        numeric = ref.predict([e],mu)[0]
        domain_error = max(domain_error,abs(value-numeric))
        assert value >= 0.
        assert oracle.predict_at([dict(e,probe=-e['probe'])],mu)[0] == value
        assert oracle.predict_at([dict(e,probe=0.)],mu)[0] == 0.
        assert oracle.predict_at([dict(e,temperature=.5)],mu)[0] == value
    assert domain_error < 1e-8
    # Independent stationary heat-kernel series. Check finite times as well
    # as the rate, so the prefactor cannot conceal a normalization error.
    kernel_error,refinement_error = 0.,0.
    for nu in [.2,.5,1.,1.5,2.,2.5]:
        for time in [.4,1.,3.,10.,14.]:
            rho = np.exp(-time)
            analytic = 2*gammaln(1+nu/2)-gammaln(1+nu)+np.log(hyp2f1(nu/2,nu/2,1+nu,rho*rho))
            numeric = ref.log_scaled_characteristic(nu,time,64)
            refined = ref.log_scaled_characteristic(nu,time,128)
            kernel_error = max(kernel_error,abs(numeric-analytic))
            refinement_error = max(refinement_error,abs(numeric-refined))
    assert kernel_error < 1e-10 and refinement_error < 1e-10
    # At integer probes only, the continuous winding characteristic equals
    # the endpoint-angle harmonic. Exact Cartesian OU draws check this
    # without discretizing paths or imposing an origin cutoff.
    checks=[]
    rng = np.random.default_rng(615017)
    for time,nu in [(.4,1),(1.,1),(.4,2),(1.,2)]:
        count=524288
        initial=rng.normal(size=(count,2))
        rho=np.exp(-time)
        final=rho*initial+np.sqrt(1-rho*rho)*rng.normal(size=(count,2))
        angle=np.arctan2(final[:,1],final[:,0])-np.arctan2(initial[:,1],initial[:,0])
        samples=np.cos(nu*angle)
        observed=float(np.mean(samples));stderr=float(np.std(samples,ddof=1)/np.sqrt(count))
        expected=float(np.exp(-nu*time+ref.log_scaled_characteristic(nu,time,128)))
        assert abs(observed-expected) < 6*stderr
        checks.append(dict(time=time,probe=nu,samples=count,observed=observed,expected=expected,
                           standard_error=stderr,standardized_error=(observed-expected)/stderr))
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_error=recovery,
                  domain_cases=len(cases),domain_reference_max_error=domain_error,
                  heat_kernel_series_max_log_error=kernel_error,
                  heat_kernel_refinement_max_log_error=refinement_error,
                  cartesian_endpoint_checks=checks,
                  invariants=['nonnegative rate','even probe symmetry','zero probe',
                              'temperature independence for positive temperature'])
    path = ROOT/'results/expansion-trapped-winding-r1-validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
