"""Calibrate and validate anchoring work against minimized full energy."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np

BASE = Path(__file__).resolve().parents[1]
TASK = BASE/'tasks/nematic-wall-torque'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual, truth):
    return float(np.linalg.norm(actual-truth)/np.linalg.norm(truth))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    oracle = load(TASK/'solution/model.py', 'oracle')
    source = load(BASE/'scripts/nematic_wall_torque_baseline.py', 'source')
    ref = load(TASK/'tests/reference.py', 'ref')
    meta = json.loads((TASK/'tests/metadata.json').read_text())
    inputs, sigma, truth_parameter = ref.calibration_inputs(), meta['sigma'], ref.TRUE_PARAMETER
    clean = ref.predict(inputs)
    if args.generate:
        rng = np.random.default_rng(meta['calibration_seed'])
        values = clean+rng.normal(0, sigma, len(inputs))
        records = [dict(input=e, value=float(v), sigma=sigma) for e, v in zip(inputs, values)]
        for path in [TASK/'environment/data/calibration.json', TASK/'tests/data/calibration.json']:
            path.write_text(json.dumps(records, indent=2)+'\n')
    records = json.loads((TASK/'tests/data/calibration.json').read_text())
    assert records == json.loads((TASK/'environment/data/calibration.json').read_text())
    assert [r['input'] for r in records] == inputs
    assert all(set(r) == {'input', 'value', 'sigma'} and r['sigma'] == .003 for r in records)
    assert meta['prediction_limit'] == .04
    hidden = ref.hidden_inputs()
    truth = {key: ref.predict(es) for key, es in hidden.items()}
    curved = [key for key in hidden if key != 'flat_anchors']

    def evaluate(module, records):
        model = module.Model().fit(records)
        values = np.array([r['value'] for r in records])
        return {
            'elastic_constant': model.elastic_constant,
            'parameter_relative_error': abs(model.elastic_constant/truth_parameter-1),
            'calibration_chi2': float(np.sum(((model.predict(inputs)-values)/sigma)**2)/(len(inputs)-1)),
            'hidden': {key: error(model.predict(es), truth[key]) for key, es in hidden.items()},
        }

    controls = {name: evaluate(module, records) for name, module in [('oracle', oracle), ('shortcut', source)]}
    noise = []
    rng = np.random.default_rng(meta['noise_validation_seed'])
    for _ in range(256):
        rr = [dict(input=e, value=float(v), sigma=sigma) for e, v in zip(inputs, clean+rng.normal(0, sigma, len(inputs)))]
        result = {name: evaluate(module, rr) for name, module in [('oracle', oracle), ('shortcut', source)]}
        for value in result.values():
            assert value['parameter_relative_error'] < .03 and value['calibration_chi2'] < 1.5
        assert max(result['oracle']['hidden'].values()) < .04
        assert min(result['shortcut']['hidden'][key] for key in curved) > .04
        noise.append(result)

    recovery = []
    for modulus in [8., 9.5, 11.7, 14., 16.]:
        rr = [dict(input=e, value=float(v), sigma=sigma) for e, v in zip(inputs, oracle.predict_at(inputs, modulus))]
        recovery.append(abs(oracle.Model().fit(rr).elastic_constant/modulus-1))
    equivalence = float(np.max(abs(oracle.predict_at(inputs, truth_parameter)-source.predict_at(inputs, truth_parameter))))
    assert equivalence == 0 and max(recovery) < 1e-14
    basis = oracle.predict_at(inputs, 1.)
    objective_curvature = float(2*np.sum((basis/sigma)**2))
    assert objective_curvature > 0

    cases = list(itertools.product([.7, 1.3], [2., 4.], [.05, .15], [.3, .7], [8., 16.]))
    rng = np.random.default_rng(241117)
    cases += [(rng.uniform(.7, 1.3), rng.uniform(2, 4), rng.uniform(.05, .15), rng.uniform(.3, .7), rng.uniform(8, 16)) for _ in range(32)]
    differences, refinement, source_checks, identities, separation, signals = [], [], [], [], [], []
    for index, (radius, ratio, low, high, modulus) in enumerate(cases):
        es = [ref.annulus(radius, ratio, low, high)]
        exact, approximate, independent = oracle.predict_at(es, modulus), source.predict_at(es, modulus), ref.predict(es, modulus)
        differences.append(error(exact, independent))
        source_checks.append(abs(approximate[0]/ref.energy_torque(radius, ratio, low, high, modulus, saddle_ratio=0)-1))
        separation.append(error(approximate, independent))
        signals.append(float(exact[0]))
        energy, detail = ref.minimized_energy(low, high, ratio, modulus)
        identities.append(abs(energy-detail['reduced_energy']))
        assert energy > 0 and detail['smallest_hessian_eigenvalue'] > 0
        if index % 8 == 0:
            refinement.append(error(ref.predict(es, modulus, modes=64), independent))
    assert max(differences) < 2e-6 and max(refinement) < 2e-6
    assert max(source_checks) < 2e-6 and min(separation) > .4 and min(signals) > .3
    assert max(identities) < 1e-12

    # In the specified coefficient convention f=K|grad n|^2/2. The branch's
    # second variation for all fixed-trace vector perturbations has this bound.
    potential_bound = (.65/np.log(2)+np.log(4)/2)**2+np.sin(.7)**2
    stability_bound = np.pi**2/np.log(4)**2-potential_bound
    assert stability_bound > 0 and np.cos(1.4) > 0
    scaling = []
    for module in [oracle, source]:
        for key in curved:
            es = hidden[key]
            scaling.append(error(module.predict_at(es, 16), 2*module.predict_at(es, 8)))
            enlarged = [dict(e, inner_radius=1.1*e['inner_radius']) for e in es]
            scaling.append(error(module.predict_at(enlarged, 11.7), module.predict_at(es, 11.7)/1.1))
    assert max(scaling) < 1e-14

    report = {
        'task': 'nematic-wall-torque', 'revision': 1, 'model_evaluations': 0,
        'controls': controls, 'noise_trials': 256,
        'noise': {'all_calibration_and_parameter_pass': True, 'oracle_passes': 256, 'shortcut_rejections': 256,
                  'max_calibration_chi2': max(r[k]['calibration_chi2'] for r in noise for k in r),
                  'max_parameter_relative_error': max(r[k]['parameter_relative_error'] for r in noise for k in r),
                  'max_oracle_hidden_error': max(v for r in noise for v in r['oracle']['hidden'].values()),
                  'min_shortcut_curved_error': min(r['shortcut']['hidden'][key] for r in noise for key in curved)},
        'physics': {'independent_domain_cases': len(cases), 'oracle_reference_relative_error_max': max(differences),
                    'reference_refinement_relative_error_max': max(refinement),
                    'shortcut_own_energy_relative_error_max': max(source_checks),
                    'full_energy_gradient_identity_error_max': max(identities),
                    'domain_shortcut_error_range': [min(separation), max(separation)],
                    'domain_signal_range': [min(signals), max(signals)],
                    'calibration_equivalence_error': equivalence, 'modulus_length_scaling_error': max(scaling),
                    'fixed_trace_second_variation_lower_bound': float(stability_bound),
                    'stability_scope': 'Local stability of the prepared smooth branch against arbitrary zero-trace vector perturbations; not uniqueness of all3D stationary states.',
                    'meridional_convexity_bound': float(np.cos(1.4)),
                    'force_sign': 'Supplied reversible work, about each local azimuthal tangent; summed vector torque about cylinder axis is not the readout.'},
        'identifiability': {'positive_linear_response_basis': True, 'weighted_objective_second_derivative': objective_curvature,
                            'max_full_range_noiseless_recovery_error': max(recovery)},
        'data_integrity': {'records': len(records), 'distinct_calibration_settings': len({json.dumps(e, sort_keys=True) for e in inputs}),
                           'fixed_instrument_sigma': sigma, 'public_private_identical': True,
                           'sigma_independent_of_response_and_parameter': True,
                           'calibration_seed': meta['calibration_seed'], 'noise_seed': meta['noise_validation_seed'], 'domain_seed': 241117},
        'runtime_seconds': time.monotonic()-start,
        'source_sha256': {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TASK.rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
    }
    (BASE/'results/nematic-wall-torque-r1-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
