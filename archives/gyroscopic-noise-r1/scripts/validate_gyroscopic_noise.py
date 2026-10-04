#!/usr/bin/env python3
"""Generate data intentionally and validate the gyroscopic-noise controls."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm, solve_continuous_lyapunov

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/gyroscopic-noise'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load('gyro_oracle', TASK/'solution/model.py')
shortcut = load('gyro_shortcut', ROOT/'scripts/gyroscopic_noise_baseline.py')
reference = load('gyro_reference', TASK/'tests/reference.py')
metadata = json.loads((TASK/'tests/metadata.json').read_text())


def records_for(values):
    return [{'input': e, 'value': float(y), 'sigma': metadata['sigma']}
            for e, y in zip(calibration, values)]


def errors(model, drag):
    return {name: float(np.sqrt(np.mean((model.predict_at(inputs, drag)-truth[name])**2)
                                /np.mean(truth[name]**2)))
            for name, inputs in hidden.items()}


def local_control(path):
    with tempfile.TemporaryDirectory(prefix='gyro-control-') as directory:
        directory = Path(directory)
        shutil.copy2(path, directory/'model.py')
        shutil.copy2(TASK/'environment/test_public.py', directory/'test_public.py')
        shutil.copytree(TASK/'environment/data', directory/'data')
        env = dict(os.environ, PYTHONPATH=str(directory), OPENBLAS_NUM_THREADS='1')
        started = time.monotonic()
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                 str(directory/'test_public.py'), str(TASK/'tests/test_hidden.py')],
                                capture_output=True, text=True, env=env)
        return {'returncode': result.returncode, 'seconds': time.monotonic()-started,
                'stdout': result.stdout, 'stderr': result.stderr}


parser = argparse.ArgumentParser()
parser.add_argument('--generate', action='store_true')
parser.add_argument('--samples', type=int, default=256)
args = parser.parse_args()
calibration = reference.calibration_inputs()*metadata['calibration_repeats']
truth_cal = reference.predict(calibration, reference.TRUE_PARAMETER)
if args.generate:
    rng = np.random.default_rng(metadata['calibration_seed'])
    records = records_for(truth_cal+metadata['sigma']*rng.normal(size=len(calibration)))
    text = json.dumps(records, indent=2)+'\n'
    for relative in ['environment/data/calibration.json', 'tests/data/calibration.json']:
        (TASK/relative).write_text(text)
records = json.loads((TASK/'tests/data/calibration.json').read_text())
assert records == json.loads((TASK/'environment/data/calibration.json').read_text())
hidden = reference.hidden_inputs()
truth = {name: reference.predict(inputs, reference.TRUE_PARAMETER) for name, inputs in hidden.items()}
report = {'task': 'gyroscopic-noise', 'revision': 1, 'calibration_records': len(records),
          'seeds': {'calibration': metadata['calibration_seed'], 'noise': metadata['noise_seed']}}
for name, module in [('oracle', oracle), ('shortcut', shortcut)]:
    model = module.Model().fit(records)
    residual = (model.predict(calibration)-np.array([r['value'] for r in records]))/metadata['sigma']
    report[name] = {'drag': model.drag, 'calibration_chi2': float(residual@residual/(len(records)-1)),
                    'hidden': errors(module, model.drag)}
report['exact_calibration_equivalence'] = float(np.max(abs(oracle.predict_at(calibration,.67)-shortcut.predict_at(calibration,.67))))
assert report['exact_calibration_equivalence'] < 2e-13
assert max(report['oracle']['hidden'].values()) < .04
assert all(report['shortcut']['hidden'][k] > .04 for k in hidden if k != 'instantaneous_and_reciprocal')
assert report['shortcut']['hidden']['instantaneous_and_reciprocal'] < .04

rng = np.random.default_rng(metadata['noise_seed'])
noise = {'samples': args.samples, 'calibration_passes': 0, 'parameter_passes': 0,
         'oracle_passes': 0, 'shortcut_rejections': 0,
         'max_chi2': 0., 'max_parameter_error': 0., 'oracle_max_error': 0.,
         'shortcut_min_discriminating_error': float('inf'), 'max_fit_difference': 0.}
for index in range(args.samples):
    noisy = records_for(truth_cal+metadata['sigma']*rng.normal(size=len(calibration)))
    good = oracle.Model().fit(noisy)
    bad = shortcut.Model().fit(noisy)
    residual = (good.predict(calibration)-np.array([r['value'] for r in noisy]))/metadata['sigma']
    chi = float(residual@residual/(len(noisy)-1))
    parameter = abs(good.drag/reference.TRUE_PARAMETER-1)
    good_errors = errors(oracle, good.drag)
    bad_errors = errors(shortcut, bad.drag)
    noise['calibration_passes'] += chi < 1.5
    noise['parameter_passes'] += parameter < .03
    noise['oracle_passes'] += max(good_errors.values()) < .04
    noise['shortcut_rejections'] += all(bad_errors[k] > .04 for k in hidden if k != 'instantaneous_and_reciprocal')
    noise['max_chi2'] = max(noise['max_chi2'], chi)
    noise['max_parameter_error'] = max(noise['max_parameter_error'], parameter)
    noise['oracle_max_error'] = max(noise['oracle_max_error'], max(good_errors.values()))
    noise['shortcut_min_discriminating_error'] = min(noise['shortcut_min_discriminating_error'], min(bad_errors[k] for k in hidden if k != 'instantaneous_and_reciprocal'))
    noise['max_fit_difference'] = max(noise['max_fit_difference'], abs(good.drag-bad.drag))
    if (index+1) % 64 == 0:
        print(f'validated {index+1}/{args.samples} noise draws', flush=True)
for key in ['calibration_passes','parameter_passes','oracle_passes','shortcut_rejections']:
    assert noise[key] == args.samples, (key, noise)
report['noise'] = {k: int(v) if isinstance(v,np.integer) else v for k,v in noise.items()}

checks = {'reference_max_abs': 0., 'spectrum_min_eigenvalue': float('inf'),
          'shortcut_min_eigenvalue': float('inf'), 'gibbs_max_abs': 0.,
          'lyapunov_max_abs': 0., 'stability_max_real': -float('inf'),
          'field_reversal_max_abs': 0., 'reflection_max_abs': 0.,
          'dc_identity_max_abs': 0., 'zero_field_max_abs': 0.}
rng = np.random.default_rng(640921)
all_cases = [(e,g) for g in [.4,.67,1.1] for inputs in hidden.values() for e in inputs]
for _ in range(128):
    e=reference.experiment(rng.uniform(0,2.2),rng.uniform(-1.6,1.6),rng.uniform(-np.pi/2,np.pi/2),
                           rng.uniform(-1.3,1.3),rng.uniform(0,3),rng.uniform(.5,1.4),
                           rng.uniform(.7,1.3),rng.uniform(1.6,2.5))
    all_cases.append((e,rng.uniform(.4,1.1)))
for e,g in all_cases:
    s=oracle.spectrum(e,g);s0=shortcut.spectrum(e,g)
    checks['reference_max_abs']=max(checks['reference_max_abs'],float(abs(s-reference.position_spectrum(e,g)).max()))
    checks['spectrum_min_eigenvalue']=min(checks['spectrum_min_eigenvalue'],float(np.linalg.eigvalsh(s).min()))
    checks['shortcut_min_eigenvalue']=min(checks['shortcut_min_eigenvalue'],float(np.linalg.eigvalsh(s0).min()))
    a,q,k=reference.dynamics(e,g)
    c=solve_continuous_lyapunov(a,-q)
    gibbs=np.zeros((4,4));gibbs[:2,:2]=e['temperature']*np.linalg.inv(k);gibbs[2:,2:]=e['temperature']*np.eye(2)
    checks['gibbs_max_abs']=max(checks['gibbs_max_abs'],float(abs(c-gibbs).max()))
    checks['lyapunov_max_abs']=max(checks['lyapunov_max_abs'],float(abs(a@c+c@a.T+q).max()))
    checks['stability_max_real']=max(checks['stability_max_real'],float(np.linalg.eigvals(a).real.max()))
    rev=dict(e,field=-e['field'])
    checks['field_reversal_max_abs']=max(checks['field_reversal_max_abs'],float(abs(oracle.spectrum(rev,g)-s.T).max()))
    refl=dict(e,field=-e['field'],trap_angle=-e['trap_angle'],weight=-e['weight'])
    checks['reflection_max_abs']=max(checks['reflection_max_abs'],float(abs(oracle.predict_at([refl],g)-oracle.predict_at([e],g))[0]))
    zero=dict(e,field=0.)
    checks['zero_field_max_abs']=max(checks['zero_field_max_abs'],float(abs(oracle.spectrum(zero,g)-shortcut.spectrum(zero,g)).max()))
    dc=dict(e,frequency=0.)
    exact=2*g*e['temperature']*np.linalg.matrix_power(k,-2)
    checks['dc_identity_max_abs']=max(checks['dc_identity_max_abs'],float(abs(oracle.spectrum(dc,g)-exact).max()))
assert checks['reference_max_abs'] < 1e-10
assert checks['spectrum_min_eigenvalue'] > 0 and checks['shortcut_min_eigenvalue'] > 0
assert checks['stability_max_real'] < 0
for k in ['gibbs_max_abs','lyapunov_max_abs','field_reversal_max_abs','reflection_max_abs','dc_identity_max_abs','zero_field_max_abs']:
    assert checks[k] < 1e-10,(k,checks[k])
report['physics_checks']=checks

# Direct time-domain detector autocorrelation checks the physical delay sign.
time_checks=[]
for e in [hidden['positive_bias'][0],hidden['negative_bias'][0]]:
    a,q,_=reference.dynamics(e,.67);c=solve_continuous_lyapunov(a,-q)
    def correlation(t):
        return (expm(a*t)@c) if t>=0 else (c@expm(-a.T*t))
    def integrand(t):
        c0=correlation(t); cp=correlation(t+e['delay']);cm=correlation(t-e['delay'])
        value=c0[0,0]+e['weight']**2*c0[1,1]+e['weight']*(cp[0,1]+cm[1,0])
        return 2*np.cos(e['frequency']*t)*value
    cutoff=120.
    value=quad(integrand,0,cutoff,points=[e['delay']],epsabs=2e-10,epsrel=2e-10,limit=300)[0]
    expected=oracle.predict_at([e],.67)[0]
    error=abs(value-expected)
    assert error<2e-9
    time_checks.append({'input':e,'direct_psd':value,'oracle_psd':expected,'absolute_error':error})
report['time_domain_delay_check']=time_checks

# DC observations make drag globally identifiable; additionally scan the full loss.
unique=reference.calibration_inputs()
scan=[]
for true_drag in [.401,.67,1.099]:
    noiseless=oracle.predict_at(calibration,true_drag)
    fitted=oracle.Model().fit(records_for(noiseless)).drag
    response=oracle.predict_at(unique,true_drag)
    grid=np.unique(np.append(np.linspace(.4,1.1,281),true_drag))
    loss=np.array([np.sum((oracle.predict_at(unique,g)-response)**2) for g in grid])
    assert abs(fitted-true_drag)<2e-7
    assert abs(grid[np.argmin(loss)]-true_drag)<1e-13
    scan.append({'true':true_drag,'fit':fitted,'grid_minimum':float(grid[np.argmin(loss)]),
                 'minimum_other_grid_loss':float(np.min(loss[grid!=true_drag]))})
report['identifiability']=scan
report['dc_gain_slope_min']=min(float(oracle.predict_at([dict(e,frequency=0.)],1.)[0]) for e in unique)
report['local_controls']={'oracle':local_control(TASK/'solution/model.py'),
                          'shortcut':local_control(ROOT/'scripts/gyroscopic_noise_baseline.py')}
assert report['local_controls']['oracle']['returncode']==0
assert '3 failed, 5 passed' in report['local_controls']['shortcut']['stdout']
path=ROOT/'results/gyroscopic-noise-validation.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
