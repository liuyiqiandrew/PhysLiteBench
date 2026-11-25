"""Validate coherent disordered transmission; --generate replaces checked-in data."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/disordered-wave-chain'


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
    started = time.monotonic()
    ref = load('ref', TASK/'tests/reference.py')
    oracle = load('oracle', TASK/'solution/model.py')
    baseline = load('baseline', ROOT/'scripts/disordered_wave_chain_baseline.py')
    strength, sigma = ref.TRUE_PARAMETER, .001
    calibration = ref.calibration_inputs()*12
    means = ref.predict(calibration, strength)
    assert max(abs(oracle.predict_at(calibration,strength)-means)) < 1e-12
    assert max(abs(baseline.predict_at(calibration,strength)-means)) < 1e-12
    if args.generate:
        rng = np.random.default_rng(613001)
        records = [dict(input=e,value=float(y),sigma=sigma)
                   for e,y in zip(calibration,means+rng.normal(0,sigma,len(means)))]
        for part in ['environment','tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
        truth = {name:dict(inputs=rows,values=ref.predict(rows,strength).tolist())
                 for name,rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',baseline)]}
    report = dict(revision=1,calibration_seed=613001,noise_seed=613009,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(calibration)-[r['value'] for r in records])/sigma
        report['controls'][name] = dict(strength=model.strength,chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2'] < 1.5 and abs(model.strength/strength-1) < .03
    rng = np.random.default_rng(613009)
    fits,chis = [],[]
    for _ in range(256):
        noisy = [dict(r,value=float(y)) for r,y in zip(records,means+rng.normal(0,sigma,len(means)))]
        model = oracle.Model().fit(noisy)
        fits.append(model.strength)
        residual = (model.predict(calibration)-[r['value'] for r in noisy])/sigma
        chis.append(float(residual@residual)/(len(noisy)-1))
    report.update(noise_draws=256,noise_parameter_range=[min(fits),max(fits)],noise_max_chi2=max(chis))
    assert max(chis) < 1.5 and max(abs(np.asarray(fits)/strength-1)) < .03
    for name,rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        fresh = ref.predict(rows,strength)
        refined = ref.predict(rows,strength,grid=2400,angles=192)
        physical = oracle.predict_at(rows,strength)
        assert np.max(abs(fresh-expected)) < 1e-12
        assert error(fresh,refined) < 1e-8 and error(physical,refined) < 1e-8
        errors = {key:error(m.predict(rows),expected) for key,m in models.items()}
        noise_oracle = max(error(oracle.predict_at(rows,p),expected) for p in fits)
        noise_shortcut = min(error(baseline.predict_at(rows,p),expected) for p in fits)
        assert noise_oracle < .04
        if name != 'shared_anchors':
            assert noise_shortcut > .04
        report['groups'][name] = dict(errors=errors,noise_oracle_max=noise_oracle,
                                     noise_shortcut_min=noise_shortcut,
                                     reference_refinement_error=error(fresh,refined),
                                     oracle_reference_error=error(physical,refined))
    # All calibration responses strictly decrease with strength. Both N=1
    # and N=2 equal 1/(1+N*b^2/k^2), so the parameter is globally identified.
    recovery_error = 0.
    for b in np.linspace(.6,1.,41):
        noiseless = [dict(r,value=float(y)) for r,y in zip(records,oracle.predict_at(calibration,b))]
        recovery_error = max(recovery_error,abs(oracle.Model().fit(noiseless).strength-b))
    assert recovery_error < 1e-7
    rng = np.random.default_rng(613013)
    cases = [(ref.experiment(int(rng.integers(0,13)),float(rng.uniform(.8,1.6))),float(rng.uniform(.6,1.)))
             for _ in range(64)]
    cases += [(ref.experiment(n,k),b) for n in [0,1,2,3,6,12] for k in [.8,1.6] for b in [.6,1.]]
    domain_error,spectral_error,three_error = 0.,0.,0.
    for e,b in cases:
        value = oracle.response(e,b)
        numeric = ref.predict([e],b)[0]
        refined = oracle.response(e,b,radial=320,angular=384,upper=16.)
        domain_error = max(domain_error,abs(value-numeric))
        spectral_error = max(spectral_error,abs(value-refined))
        assert 0 <= value <= 1.+1e-12
        assert oracle.response(dict(e,count=0),b) == 1.
        lam = (b/e['wave_number'])**2
        expected = 1/((1+lam)*np.sqrt(1+4*lam))
        three_error = max(three_error,abs(oracle.response(dict(e,count=3),b)-expected))
    assert domain_error < 1e-8 and spectral_error < 1e-10 and three_error < 1e-10
    # Direct complex delta-barrier transfer products independently check the
    # underlying coherent process. These sampling estimates are not truth.
    microscopic=[]
    for i,(b,k,n) in enumerate([(.6,1.6,3),(.83,.8,4),(.83,1.1,6),(.83,1.4,8),(1.,.8,12),(1.,1.6,12)]):
        samples = ref.transfer_samples(b,k,n,samples=262144,seed=613017+i)
        expected = oracle.response(ref.experiment(n,k),b)
        observed = float(np.mean(samples))
        stderr = float(np.std(samples,ddof=1)/np.sqrt(len(samples)))
        assert max(samples) <= 1.+1e-10 and min(samples) > 0.
        assert abs(observed-expected) < 6*stderr
        log_error = float(np.mean(np.log(samples))-n*np.log(1/(1+(b/k)**2)))
        log_stderr = float(np.std(np.log(samples),ddof=1)/np.sqrt(len(samples)))
        assert abs(log_error) < 6*log_stderr
        microscopic.append(dict(strength=b,wave_number=k,count=n,samples=len(samples),
                                observed=observed,expected=expected,standard_error=stderr,
                                standardized_error=(observed-expected)/stderr,
                                mean_log_error=log_error,mean_log_standard_error=log_stderr))
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_absolute_error=recovery_error,
                  domain_cases=len(cases),domain_reference_max_absolute_difference=domain_error,
                  spectral_refinement_max_absolute_difference=spectral_error,
                  three_barrier_identity_max_absolute_error=three_error,
                  spectral_tail_bound=float(4*(14+1/np.pi)*np.exp(-14*np.pi)),
                  microscopic_checks=microscopic,
                  invariants=['zero-barrier unit transmission','one/two-barrier exact limits',
                              'three-barrier exact identity','passive transmission bounds',
                              'additive disorder-averaged log transmission'],
                  wall_seconds=time.monotonic()-started)
    (ROOT/'results/expansion-disordered-wave-chain-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
