"""Validate pairwise gas friction against independent reduced equations."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("OMP_NUM_THREADS","1")
import time
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/ternary-gas'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    args = parser.parse_args(); start=time.time()
    oracle = module('gas_oracle', TASK/'solution/model.py')
    shortcut = module('gas_shortcut', ROOT/'scripts/ternary_gas_baseline.py')
    ref = module('gas_reference', TASK/'tests/reference.py')
    if args.generate:
        x = np.linspace(0, 1, 25)
        t = np.array([.2, .5, 1., 2., 3., 5., 8., 12., 18., 26.])
        profiles = []
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            initial = np.zeros((len(x), 3))
            initial[:, a] = .5+.3*np.cos(np.pi*x)
            initial[:, b] = 1-initial[:, a]
            profiles.append(initial)
        mean = np.array([ref.predict(t, x, p, ref.TRUE_DIFFUSIVITY, refinement=8)
                         for p in profiles])
        sigma = np.full_like(mean, .002)
        measured = mean+np.random.default_rng(9400).normal(size=mean.shape)*sigma
        path = TASK/'environment/data/calibration.npz'
        np.savez(path, t=t, x=x, initial=profiles, mole_fraction=measured, sigma=sigma)
        shutil.copyfile(path, TASK/'tests/data/calibration.npz')
    assert (TASK/'environment/data/calibration.npz').read_bytes() == (TASK/'tests/data/calibration.npz').read_bytes()
    data = dict(np.load(TASK/'environment/data/calibration.npz'))
    result = {'revision': 2, 'calibration_seed': 9400, 'noise_seed': 19400,
              'noise_trials': args.noise_trials, 'controls': {}}
    for name, implementation in [('oracle', oracle), ('shortcut', shortcut)]:
        model = implementation.Model().fit(data)
        measured = ref.metrics(model, data)
        result['controls'][name] = measured
        assert measured['calibration_chi2'] < 1.5 and measured['parameter_relative_error'] < .03
        assert all((v < .04) == (name == 'oracle') for v in measured['hidden'].values())
    a, b = oracle.Model(), shortcut.Model()
    a.diffusivity = b.diffusivity = ref.TRUE_DIFFUSIVITY
    equivalence = max(np.max(np.abs(a.predict(data['t'], data['x'], p)-
                                    b.predict(data['t'], data['x'], p)))
                      for p in data['initial'])
    assert equivalence < 1e-12
    mean = np.array([ref.predict(data['t'], data['x'], p, ref.TRUE_DIFFUSIVITY, refinement=8)
                     for p in data['initial']])
    rng = np.random.default_rng(19400)
    fitted, chi = [], []
    for i in range(args.noise_trials):
        noisy = {**data, 'mole_fraction': mean+rng.normal(size=mean.shape)*data['sigma']}
        model = oracle.Model().fit(noisy)
        control = shortcut.Model().fit(noisy)
        assert abs(model.diffusivity-control.diffusivity)<1e-12
        pred = np.array([model.predict(data['t'], data['x'], p) for p in data['initial']])
        fitted.append(model.diffusivity)
        chi.append(float(np.sum(((pred-noisy['mole_fraction'])/data['sigma'])**2)/(pred.size-1)))
    assert max(chi) < 1.5
    assert max(abs(np.array(fitted)/ref.TRUE_DIFFUSIVITY-1)) < .03
    separation = {name: [] for name in ['oracle', 'shortcut']}
    mass_error = 0.
    convergence = []
    entropy_increase = 0.
    quadratic_increase=0.; minimum_fraction=1.; sum_error=0.; oracle_reference=[]
    for t, x, initial in ref.hidden_inputs():
        truth = ref.predict(t, x, initial, ref.TRUE_DIFFUSIVITY, refinement=4)
        finer = ref.predict(t, x, initial, ref.TRUE_DIFFUSIVITY, refinement=8)
        convergence.append(float(np.max(abs(truth-finer))))
        oracle_reference.append(float(np.max(abs(a.predict(t,x,initial)-truth))))
        for name, implementation in [('oracle', oracle), ('shortcut', shortcut)]:
            for d in [min(fitted), max(fitted)]:
                model = implementation.Model()
                model.diffusivity = d
                pred = model.predict(t, x, initial)
                avg = np.trapezoid(initial, x, axis=0)
                error = np.sqrt(np.mean((pred-truth)**2))/np.sqrt(np.mean((truth-avg)**2))
                separation[name].append(float(error))
                mass_error = max(mass_error, float(np.max(abs(np.trapezoid(pred, x, axis=1)-avg))))
                minimum_fraction=min(minimum_fraction,float(pred.min()))
                sum_error=max(sum_error,float(np.max(abs(pred.sum(axis=2)-1))))
                assert pred.min() >= -1e-10 and np.max(abs(pred.sum(axis=2)-1)) < 1e-9
                if name=='shortcut':
                    quadratic=np.trapezoid(np.sum((pred-avg)**2/avg,axis=2)/2,x,axis=1)
                    quadratic_increase=max(quadratic_increase,float(np.max(np.diff(quadratic))))
                assert (error < .04) == (name == 'oracle')
                if name == 'oracle':
                    entropy = np.trapezoid(np.sum(pred*np.log(pred), axis=2), x, axis=1)
                    entropy_increase = max(entropy_increase, float(np.max(np.diff(entropy))))
    assert mass_error < 1e-8 and max(convergence) < .0001 and entropy_increase < 1e-8 and quadratic_increase<1e-8
    t,x,initial=ref.hidden_inputs()[2]
    mean_composition=np.trapezoid(initial,x,axis=0)
    small=mean_composition+1e-3*(initial-mean_composition)
    linearization_error=float(np.max(abs(a.predict(t,x,small)-b.predict(t,x,small))))
    uniform=np.tile(mean_composition,(len(x),1))
    stationary_error=max(float(np.max(abs(m.predict(t,x,uniform)-uniform))) for m in [a,b])
    reversal=max(float(np.max(abs(m.predict(t,x,initial)-m.predict(t,x,initial[::-1])[:,::-1]))) for m in [a,b])
    assert linearization_error<1e-7 and stationary_error<1e-12 and reversal<1e-7
    result['physical_checks'] = {'binary_shortcut_equivalence': float(equivalence),
                                 'independent_mesh_max_difference': max(convergence),
                                 'molar_conservation_max_error': mass_error,
                                 'mixing_free_energy_increase_max': entropy_increase,
                                 'shortcut_quadratic_free_energy_increase_max':quadratic_increase,
                                 'minimum_fraction_both_controls':minimum_fraction,'fraction_sum_error':sum_error,
                                 'oracle_reference_max_error':max(oracle_reference),
                                 'weak_perturbation_linearization_error':linearization_error,
                                 'uniform_stationarity_error':stationary_error,'spatial_reversal_error':reversal}
    result['noise'] = {'calibration_chi2_max': max(chi), 'calibration_pass_fraction': 1.,
                       'parameter_min': min(fitted), 'parameter_max': max(fitted),
                       'parameter_relative_error_max': float(max(abs(np.array(fitted)/ref.TRUE_DIFFUSIVITY-1)))}
    result['hidden_parameter_extrema_check'] = {k: {'min': min(v), 'max': max(v)} for k, v in separation.items()}
    result['seconds']=time.time()-start
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'environment/data/calibration.npz',ROOT/'scripts/ternary_gas_baseline.py',Path(__file__)]}
    output = ROOT/'jobs/ternary-gas-r2-validation'
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
