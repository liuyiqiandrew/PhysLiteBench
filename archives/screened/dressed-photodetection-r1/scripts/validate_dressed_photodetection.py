"""Check detector transition strengths independently of the Gaussian solution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/dressed-photodetection'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(generate=False, noise_trials=256):
    started = time.time()
    good = load(TASK/'solution/model.py', 'oracle')
    bad = load(ROOT/'scripts/dressed_photodetection_baseline.py', 'shortcut')
    ref = load(TASK/'tests/reference.py', 'reference')
    inputs = ref.calibration_inputs()
    true = ref.TRUE_PARAMETER
    sigma = 2e-6
    clean = ref.predict(inputs, true)
    def records(values):
        return [dict(input=e, value=float(v), sigma=sigma) for e, v in zip(inputs, values)]
    if generate:
        data = json.dumps(records(clean+np.random.default_rng(948101).normal(0, sigma, len(inputs))), indent=2)+'\n'
        for file in ['environment/data/calibration.json', 'tests/data/calibration.json']:
            (TASK/file).write_text(data)
    data = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert data == json.loads((TASK/'tests/data/calibration.json').read_text())
    groups = ref.hidden_inputs()
    truth = {name: ref.predict(es, true) for name, es in groups.items()}
    def errors(module, parameter):
        return {name: float(np.linalg.norm(parameter*np.array([module.response(e) for e in es])-truth[name])/np.linalg.norm(truth[name])) for name, es in groups.items()}
    controls = {}
    for label, module in [('oracle', good), ('shortcut', bad)]:
        model = module.Model().fit(data)
        res = (model.predict(inputs)-np.array([r['value'] for r in data]))/sigma
        controls[label] = dict(detection_rate=model.detection_rate, calibration_chi2=float(res@res)/(len(data)-1), hidden=errors(module, model.detection_rate))
        assert abs(model.detection_rate/true-1) < .03 and controls[label]['calibration_chi2'] < 1.5
        assert (max(controls[label]['hidden'].values()) < .03 if label == 'oracle' else min(controls[label]['hidden'].values()) > .03)
    equivalence = max(abs(good.response(e)-bad.response(e)) for e in inputs)
    hidden_error = hidden_refinement = covariance_error = 0.
    for es in groups.values():
        for e in es:
            coarse = ref.fock_response(e, 36)
            fine = ref.fock_response(e, 44)
            hidden_error = max(hidden_error, abs(good.response(e)-coarse))
            hidden_refinement = max(hidden_refinement, abs(coarse-fine))
            energy, _, bare = ref.spectrum(e['frequency'], e['interaction'], 44)
            p = np.exp(-energy/e['temperature']); p /= p.sum()
            covariance_error = max(covariance_error, abs(bad.response(e)-(p@bare-bare[0])))
    corners = []; corner_error = corner_refinement = 0.; ground_rate = 0.
    for frequency in [1.1, 1.8]:
        for interaction in [0., .4*np.sqrt(frequency)]:
            for temperature in [.12, .55]:
                e = ref.experiment(frequency, interaction, temperature)
                f44 = ref.fock_response(e, 44); f52 = ref.fock_response(e, 52)
                corner_error = max(corner_error, abs(good.response(e)-f52))
                corner_refinement = max(corner_refinement, abs(f52-f44))
                corners.append(dict(input=e, oracle=good.response(e), reference52=f52, error44to52=abs(f44-f52)))
            _, weights, _ = ref.spectrum(frequency, interaction, 44)
            ground_rate = max(ground_rate, abs(weights[0]))
    frequency_limits = []; sign_error = 0.; minimum_response = np.inf; minimum_stiffness = np.inf
    for frequency in np.linspace(1.1, 1.8, 9):
        for ratio in np.linspace(-.4, .4, 9):
            interaction = ratio*np.sqrt(frequency)
            stiffness = np.array([[1., 2*interaction*np.sqrt(frequency)], [2*interaction*np.sqrt(frequency), frequency**2]])
            w = np.sqrt(np.linalg.eigvalsh(stiffness)); frequency_limits.extend(w.tolist()); minimum_stiffness = min(minimum_stiffness, w[0]**2)
            for temperature in [.12, .55]:
                e = ref.experiment(frequency, interaction, temperature)
                sign_error = max(sign_error, abs(good.response(e)-good.response(dict(e, interaction=-interaction))))
                minimum_response = min(minimum_response, good.response(e), bad.response(e))
    parameter_recovery = max(abs(bad.Model().fit(records(ref.predict(inputs, p))).detection_rate/p-1) for p in [.008, .014, .02])
    assert equivalence < 1e-15 and hidden_error < 1e-7 and hidden_refinement < 1e-7 and covariance_error < 1e-9
    assert corner_error < 1e-7 and corner_refinement < 1e-7 and ground_rate < 1e-15
    assert min(frequency_limits) > .2 and max(frequency_limits) < 2.5 and minimum_stiffness > 0
    assert minimum_response > 0 and sign_error < 1e-15 and parameter_recovery < 1e-14
    rng = np.random.default_rng(948103); noise = []
    for _ in range(noise_trials):
        sample = records(clean+rng.normal(0, sigma, len(inputs)))
        model = bad.Model().fit(sample)
        res = (model.predict(inputs)-np.array([r['value'] for r in sample]))/sigma
        noise.append(dict(detection_rate=model.detection_rate, chi2=float(res@res)/(len(inputs)-1), oracle=errors(good, model.detection_rate), shortcut=errors(bad, model.detection_rate)))
    summary = dict(calibration_passes=sum(n['chi2']<1.5 for n in noise), parameter_passes=sum(abs(n['detection_rate']/true-1)<.03 for n in noise), oracle_passes=sum(max(n['oracle'].values())<.03 for n in noise), shortcut_passes=sum(max(n['shortcut'].values())<.03 for n in noise), maximum_oracle_error=max(max(n['oracle'].values()) for n in noise), minimum_shortcut_error=min(min(n['shortcut'].values()) for n in noise), maximum_chi2=max(n['chi2'] for n in noise), parameter_range=[min(n['detection_rate'] for n in noise), max(n['detection_rate'] for n in noise)])
    assert summary['calibration_passes'] == noise_trials and summary['parameter_passes'] == noise_trials and summary['oracle_passes'] == noise_trials and summary['shortcut_passes'] == 0
    files = [p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[ROOT/'scripts/dressed_photodetection_baseline.py', Path(__file__).resolve()]
    return dict(revision=1, noise_trials=noise_trials, calibration_seed=948101, noise_seed=948103, measurement_sigma=sigma, controls=controls, calibration_equivalence=equivalence, direct_fock_hidden_error=hidden_error, hidden_fock36_to44_error=hidden_refinement, direct_fock_excess_occupation_error=covariance_error, corner_fock52_error=corner_error, corner_fock44_to52_error=corner_refinement, corner_checks=corners, ground_absorption_rate=ground_rate, normal_frequency_range=[min(frequency_limits), max(frequency_limits)], minimum_stiffness=minimum_stiffness, minimum_response=minimum_response, interaction_sign_error=sign_error, parameter_recovery_error=parameter_recovery, noise=summary, noise_realizations=noise, seconds=time.time()-started, source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--generate', action='store_true'); parser.add_argument('--noise-trials', type=int, default=256); args = parser.parse_args()
    report = run(args.generate, args.noise_trials)
    (ROOT/'results/dressed-photodetection-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['noise_realizations', 'source_sha256', 'corner_checks']}, indent=2))
