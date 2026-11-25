"""Validate pulsed kinetic gas r1; --generate intentionally replaces simulated data."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/pulsed-kinetic-gas'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rms_error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    reference = load('reference', TASK/'tests/reference.py')
    oracle = load('oracle', TASK/'solution/model.py')
    shortcut = load('shortcut', ROOT/'scripts/pulsed_kinetic_gas_baseline.py')
    gamma = reference.TRUE_PARAMETER
    calibration = reference.calibration_inputs()*2
    mean = reference.predict(calibration, gamma)
    assert np.max(np.abs(oracle.predict_at(calibration, gamma)-mean)) < 1e-9
    assert np.max(np.abs(shortcut.predict_at(calibration, gamma)-mean)) < 1e-9
    if args.generate:
        rng = np.random.default_rng(611017)
        records = [dict(input=e, value=float(y), sigma=.0003)
                   for e,y in zip(calibration, mean+rng.normal(0., .0003, len(mean)))]
        encoded = json.dumps(records, indent=2)+'\n'
        for name in ['environment', 'tests']:
            (TASK/name/'data/calibration.json').write_text(encoded)
        truth = {name: dict(inputs=rows, values=reference.predict(rows, gamma).tolist())
                 for name,rows in reference.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth, indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    limits = []
    noise = np.random.default_rng(611019)
    fits = []
    chi2 = []
    for _ in range(256):
        sampled = [dict(input=e, value=float(y), sigma=.0003)
                   for e,y in zip(calibration, mean+noise.normal(0., .0003, len(mean)))]
        fitted = oracle.Model().fit(sampled)
        fits.append(fitted.friction)
        residual = (fitted.predict(calibration)-[r['value'] for r in sampled])/.0003
        chi2.append(float(residual@residual)/(len(sampled)-1))
    assert max(abs(np.asarray(fits)/gamma-1)) < .03
    # Both controls have identical likelihood throughout this calibration subset.
    for p in np.linspace(.08, .4, 33):
        assert np.max(np.abs(oracle.predict_at(calibration,p)-shortcut.predict_at(calibration,p))) < 1e-12
    # Noiseless parameter profile has a unique minimum; all signals lie on the
    # same side of their value at gamma on either side of gamma for short delays.
    short = [e for e in calibration if e['after'] <= 1.]
    curves = np.array([oracle.predict_at(short,p) for p in np.linspace(.08,.4,65)])
    assert np.all(np.diff(curves, axis=0) > 0.)
    report = {'revision':1, 'calibration_seed':611017, 'noise_seed':611019,
              'noise_draws':256, 'noise_parameter_range':[min(fits),max(fits)],
              'noise_max_relative_parameter_error':float(max(abs(np.asarray(fits)/gamma-1))),
              'noise_max_chi2':max(chi2), 'groups':{}}
    actual_models = {name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    report['controls'] = {}
    for name,model in actual_models.items():
        residual=(model.predict(calibration)-[r['value'] for r in records])/.0003
        report['controls'][name] = {'friction':model.friction,'chi2':float(residual@residual)/(len(records)-1)}
        assert report['controls'][name]['chi2'] < 1.5
    for name, rows in reference.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        fresh = reference.predict(rows, gamma)
        refined = reference.predict(rows, gamma, nx=512, nz=4097)
        assert np.max(np.abs(fresh-expected)) < 1e-12
        assert np.max(np.abs(fresh-refined)) < 1e-8
        assert np.max(np.abs(oracle.predict_at(rows,gamma)-refined)) < 1e-8
        errors = {key:rms_error(model.predict(rows),expected) for key,model in actual_models.items()}
        noise_oracle=max(rms_error(oracle.predict_at(rows,p),expected) for p in fits)
        noise_shortcut=min(rms_error(shortcut.predict_at(rows,p),expected) for p in fits)
        assert noise_oracle < .04
        if name != 'single_impulse':
            assert noise_shortcut > .04
        report['groups'][name] = dict(errors=errors, noise_oracle_max=noise_oracle,
                                      noise_shortcut_min=noise_shortcut,
                                      reference_refinement_max=float(np.max(np.abs(fresh-refined))),
                                      min_abs_signal=float(np.min(np.abs(expected))))
    rng=np.random.default_rng(611021)
    cases=[reference.experiment(thermal_speed=float(rng.uniform(.5,1)),
                               a1=float(rng.uniform(-1,1)),a2=float(rng.uniform(-1,1)),
                               k1=int(rng.integers(1,4)),k2=int(rng.integers(1,4)),
                               mode=int(rng.integers(1,4)),phase1=float(rng.uniform(-np.pi,np.pi)),
                               phase2=float(rng.uniform(-np.pi,np.pi)),delay=float(rng.uniform(0,3.5)),
                               after=float(rng.uniform(.2,3.5)),component=['sin','cos'][i%2])
           for i in range(24)]
    max_difference=0.
    for p in [.08,.17,.4]:
        computed=oracle.predict_at(cases,p)
        independent=reference.predict(cases,p,nx=512,nz=4097)
        max_difference=max(max_difference,float(np.max(abs(computed-independent))))
        assert np.all(abs(computed)<=1.+1e-12)
        assert np.max(abs(computed-independent)) < 1e-8
    report['domain_cases']=72
    report['domain_reference_max_absolute_difference']=max_difference
    report['wall_seconds']=time.monotonic()-started
    output=ROOT/'results/gpt6-pulsed-kinetic-gas-r1-validation.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
