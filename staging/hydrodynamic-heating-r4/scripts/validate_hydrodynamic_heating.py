"""Validate the r4 apparatus; read inherited calibration without regenerating it."""
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
import traceback
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

BASE = Path(__file__).resolve().parents[1]
TASK = BASE/'tasks/hydrodynamic-heating'
RESULTS = BASE/'results'
REPORT = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load('hydro_r4_oracle', TASK/'solution/model.py')
source = load('hydro_r4_source', BASE/'scripts/hydrodynamic_heating_baseline.py')
ref = load('hydro_r4_reference', TASK/'tests/reference.py')
meta = json.loads((TASK/'tests/metadata.json').read_text())
LOADED = {str(p.relative_to(BASE)): digest(p) for p in sorted(TASK.rglob('*'))
          if p.is_file() and '__pycache__' not in p.parts}
LOADED.update({str(p.relative_to(BASE)): digest(p) for p in
               [Path(__file__), BASE/'scripts/hydrodynamic_heating_baseline.py']})
KEYS = ('frequency', 'thickness', 'friction', 'relaxation')
READOUT = {'mean': 0, 'in_phase': 1, 'quadrature': 2}


def error(actual, truth):
    return float(np.linalg.norm(np.asarray(actual)-truth)/np.linalg.norm(truth))


def records(inputs, values, sigma):
    return [dict(input=e, value=float(v), sigma=sigma) for e, v in zip(inputs, values)]


def vector(mod, e, p):
    return mod.predict_at([dict(e, readout=r) for r in READOUT], p)


def metrics(mod, rows, groups, truths):
    fitted = mod.Model().fit(rows)
    residual = (fitted.predict([r['input'] for r in rows])-
                np.array([r['value'] for r in rows]))/np.array([r['sigma'] for r in rows])
    return dict(parameter=fitted.plasma_frequency,
                relative_parameter_error=abs(fitted.plasma_frequency/ref.TRUE_PARAMETER-1),
                calibration_chi2=float(residual@residual)/(len(rows)-1),
                hidden={k: error(fitted.predict(es), truths[k]) for k, es in groups.items()})


def optical_balance(e, p):
    fields = oracle.modal_fields(e, p)
    d, w, alpha, tau = (e[k] for k in ('thickness', 'frequency', 'friction', 'relaxation'))
    electric, _, current, _ = fields([0, d])
    traction = alpha*current/(1-1j*w*tau)
    contact_means = abs(traction)**2/(alpha*p*p)
    x, weights = leggauss(80)
    _, _, j, jz = fields((x+1)*d/2)
    bulk = d/2*np.dot(weights, oracle.GAMMA*abs(j)**2+oracle.NU*abs(jz)**2)/(p*p)
    absorption = 1-abs(electric[0]-1)**2-abs(electric[1])**2
    return dict(absorption=float(absorption), bulk=float(bulk),
                contact_means=contact_means.tolist(),
                residual=float(abs(absorption-bulk-sum(contact_means))))


def local_controls():
    result = {}
    for label, path in [('oracle', TASK/'solution/model.py'),
                        ('shortcut', BASE/'scripts/hydrodynamic_heating_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='hydro-r4-local-') as temp:
            p = Path(temp)
            shutil.copytree(TASK/'environment', p/'app')
            shutil.copytree(TASK/'tests', p/'tests')
            shutil.copy2(path, p/'app/model.py')
            env = os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(p/'app'),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            start = time.monotonic()
            run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                  str(p/'app/test_public.py'), str(p/'tests/test_hidden.py')],
                                 cwd=p, env=env, capture_output=True, text=True, timeout=60)
            result[label] = dict(returncode=run.returncode, seconds=time.monotonic()-start,
                                 stdout=run.stdout, stderr=run.stderr)
    path = RESULTS/'hydrodynamic-heating-r4-local-controls.json'
    assert not path.exists(), 'Preserve previous local-control evidence.'
    path.write_text(json.dumps(result, indent=2)+'\n')
    assert result['oracle']['returncode'] == 0 and '7 passed' in result['oracle']['stdout']
    assert result['shortcut']['returncode'] == 1 and '3 failed, 4 passed' in result['shortcut']['stdout']
    return {k: {n: v[n] for n in ['returncode', 'seconds']} for k, v in result.items()}


def run():
    start = time.monotonic()
    sigma = meta['measurement_sigma']
    rows = json.loads((TASK/'environment/data/calibration.json').read_text())
    original = BASE/'prototype/r3_calibration.json'
    assert (TASK/'environment/data/calibration.json').read_bytes() == original.read_bytes()
    assert (TASK/'tests/data/calibration.json').read_bytes() == original.read_bytes()
    settings = ref.calibration_inputs()
    assert [r['input'] for r in rows] == settings*48 and len(rows) == 288
    assert all(set(r) == {'input', 'value', 'sigma'} and r['sigma'] == .0002 for r in rows)
    assert (BASE/'scripts/hydrodynamic_heating_baseline.py').read_bytes() == (BASE/'prototype/r3_oracle.py').read_bytes()
    old = json.loads((BASE/'prototype/r3_validation.json').read_text())
    groups = ref.hidden_inputs()
    t0 = time.monotonic()
    truths = {k: ref.predict(es) for k, es in groups.items()}
    REPORT.update(status='in_progress', model_runs=0, docker_runs=0,
                  loaded_source_sha256=LOADED, cold_reference_seconds=time.monotonic()-t0,
                  calibration_records=len(rows), distinct_calibration_settings=len(settings),
                  fixed_sigma=sigma, calibration_sha256=digest(original),
                  calibration_regenerated=False)
    REPORT['actual_data'] = {name: metrics(mod, rows, groups, truths)
                            for name, mod in [('oracle', oracle), ('shortcut', source)]}
    assert REPORT['actual_data']['oracle']['parameter'] == REPORT['actual_data']['shortcut']['parameter']
    assert REPORT['actual_data']['oracle']['parameter'] == old['actual_data']['oracle']['parameter']
    noise = []
    for oldrow in old['noise']['rows']:
        previous = oldrow['oracle']
        p = previous['parameter']
        common = {k: previous[k] for k in ['parameter', 'relative_parameter_error', 'calibration_chi2']}
        a = dict(common, hidden={k: error(oracle.predict_at(es, p), truths[k]) for k, es in groups.items()})
        b = dict(common, hidden={k: error(source.predict_at(es, p), truths[k]) for k, es in groups.items()})
        assert a['relative_parameter_error'] < .03 and a['calibration_chi2'] < 1.5
        assert max(a['hidden'].values()) < .04 and b['hidden']['mean_anchor'] < .04
        assert all(b['hidden'][k] > .04 for k in groups if k != 'mean_anchor')
        noise.append(dict(seed_index=oldrow['seed_index'], oracle=a, shortcut=b))
    assert len(noise) == 256
    REPORT['noise'] = dict(reused_fits=256, fresh_draws=0, oracle_passes=256, shortcut_rejections=256,
        origin='prototype/r3_validation.json', origin_sha256=digest(BASE/'prototype/r3_validation.json'),
        qualification='Exact common mean map and objective for all p transfer the existing oracle fits and calibration metrics. These are the same256 realizations, with new whole-contact readouts evaluated here.',
        maximum_oracle_error=max(max(r['oracle']['hidden'].values()) for r in noise),
        minimum_shortcut_diagnostic_error=min(r['shortcut']['hidden'][k] for r in noise for k in groups if k != 'mean_anchor'), rows=noise)

    # Recheck the whole parameter interval on the actual new source, without claiming an interval proof.
    grid = np.linspace(.85, 1.15, 1001)
    curves = np.array([oracle.predict_at(settings, float(p)) for p in grid])
    source_curves = np.array([source.predict_at(settings, float(p)) for p in grid])
    assert np.array_equal(curves, source_curves)
    derivatives = np.diff(curves, axis=0)/(grid[1]-grid[0])
    fits, margins = [], []
    for p in np.linspace(.85, 1.15, 41):
        p = float(p)
        values = oracle.predict_at(settings, p)
        sample = records(settings, values, sigma)
        estimates = [m.Model().fit(sample).plasma_frequency for m in [oracle, source]]
        profile = np.sum((curves-values)**2, axis=1)
        minima = int(np.sum((profile[1:-1] < profile[:-2]) & (profile[1:-1] < profile[2:])))
        fits.append(dict(truth=p, estimates=estimates, maximum_error=max(abs(q-p) for q in estimates),
                         sampled_interior_objective_minima=minima))
        margins.append(dict(plasma_frequency=p, groups={k: error(source.predict_at(es, p), oracle.predict_at(es, p))
                                                       for k, es in groups.items()}))
    REPORT['identifiability'] = dict(qualification='Sampled full-range monotonic curves, noiseless objective profiles and endpoint/interior fit recovery; not an interval proof or a claim about arbitrary noisy objectives.',
        parameter_grid=grid.tolist(), calibration_curves=curves.tolist(),
        minimum_sampled_secant_derivative=float(derivatives.min()), exact_source_mean_equality=True,
        noiseless_fits=fits, full_parameter_margins=margins)
    assert derivatives.min() > .05 and max(r['maximum_error'] for r in fits) < 1e-7
    assert max(r['sampled_interior_objective_minima'] for r in fits) <= 1
    assert min(r['groups'][k] for r in margins for k in groups if k != 'mean_anchor') > .25

    # Refine each distinct graded/calibration setting at the true parameter and both parameter endpoints.
    controls = {tuple(e[k] for k in KEYS): dict(e, readout='mean')
                for es in groups.values() for e in es}
    controls.update({tuple(e[k] for k in KEYS): e for e in settings})
    refined = {}
    refinements = []
    for key, e in controls.items():
        for p in [.85, ref.TRUE_PARAMETER, 1.15]:
            base = ref.first_law_reference(e, p)
            variants = {}
            for label, kw in [('field', dict(field_tol=1e-10)),
                              ('time', dict(time_tol=2e-10, steps_per_period=128)),
                              ('settling', dict(settle_periods=12)),
                              ('measurement_periods', dict(periods=2))]:
                value = ref.first_law_reference(e, p, **kw)
                variants[label] = dict(values=value['values'], maximum_change=float(np.max(abs(np.array(value['values'])-base['values']))))
            pred = vector(oracle, e, p)
            row = dict(input=e, plasma_frequency=p, reference=base, oracle=pred.tolist(),
                       maximum_oracle_error=float(np.max(abs(pred-base['values']))), variants=variants)
            refined[(key, p)] = row
            refinements.append(row)
    REPORT['reference_refinement'] = dict(distinct_settings=len(controls), parameter_values=[.85, ref.TRUE_PARAMETER, 1.15],
        scope='Every graded setting and all6 calibration settings, each at all3 listed parameters; field, time, settling and measurement-period variations are separate.',
        maximum_error=max(r['maximum_oracle_error'] for r in refinements),
        maximum_change=max(v['maximum_change'] for r in refinements for v in r['variants'].values()), rows=refinements)
    assert REPORT['reference_refinement']['maximum_error'] < 1e-8
    assert REPORT['reference_refinement']['maximum_change'] < 1e-8
    scored = []
    for name, es in groups.items():
        for e in es:
            i = READOUT[e['readout']]
            row = refined[(tuple(e[k] for k in KEYS), ref.TRUE_PARAMETER)]
            scored.append(dict(group=name, input=e, reference=row['reference']['values'][i], oracle=row['oracle'][i],
                               maximum_component_refinement=max(abs(v['values'][i]-row['reference']['values'][i]) for v in row['variants'].values())))
    REPORT['scored_reference'] = scored
    REPORT['calibration_reference'] = [dict(input=e, plasma_frequency=p,
        bias_sigma=abs(refined[(tuple(e[k] for k in KEYS), p)]['reference']['values'][0]-oracle.predict_at([e], p)[0])/sigma)
        for e in settings for p in [.85, ref.TRUE_PARAMETER, 1.15]]

    cases = [(ref.experiment(w, d, a, t), p) for p, w, d, a, t in
             product([.85, 1.15], [.7, 1.5], [.3, 1.2], [.08, .4], [.2, 1.2])]
    rng = np.random.default_rng(194173)
    cases += [(ref.experiment(rng.uniform(.7, 1.5), rng.uniform(.3, 1.2), rng.uniform(.08, .4), rng.uniform(.2, 1.2)),
               float(rng.uniform(.85, 1.15))) for _ in range(32)]
    domain = []
    for e, p in cases:
        prediction, wrong = vector(oracle, e, p), vector(source, e, p)
        baseline = ref.first_law_reference(e, p)
        fine = ref.first_law_reference(e, p, field_tol=1e-10, time_tol=2e-10, steps_per_period=128)
        field_error = float(np.max(abs(oracle.modal_fields(e, p)([0, e['thickness']])-
                                      ref.bvp_fields(e, p)([0, e['thickness']]))))
        q = .5*e['frequency']*e['relaxation']
        domain.append(dict(input=e, plasma_frequency=p, oracle=prediction.tolist(), source=wrong.tolist(),
            reference=baseline, reference_error=float(np.max(abs(prediction-baseline['values']))),
            refinement=float(np.max(abs(np.array(fine['values'])-baseline['values']))), field_error=field_error,
            paired_gap=error(wrong[1:], prediction[1:]), analytic_paired_gap=q/np.sqrt(1+q*q),
            paired_signal_rms=float(np.linalg.norm(prediction[1:])/np.sqrt(2)),
            optical_balance=optical_balance(e, p), mean_equality=abs(float(prediction[0]-wrong[0]))))
    REPORT['domain'] = dict(corners=32, interiors=32, random_seed=194173, rows=domain,
        maximum_reference_error=max(r['reference_error'] for r in domain),
        maximum_refinement=max(r['refinement'] for r in domain),
        maximum_energy_residual=max(r['optical_balance']['residual'] for r in domain),
        minimum_pair_signal_rms=min(r['paired_signal_rms'] for r in domain),
        minimum_pair_gap=min(r['paired_gap'] for r in domain))
    assert REPORT['domain']['maximum_reference_error'] < 1e-8 and REPORT['domain']['maximum_refinement'] < 1e-8
    assert REPORT['domain']['maximum_energy_residual'] < 1e-10
    assert max(r['field_error'] for r in domain) < 1e-8
    assert max(r['reference']['closure_error'] for r in domain) < 1e-12
    assert max(r['reference']['energy_integral_error'] for r in domain) < 1e-8
    assert max(abs(r['paired_gap']-r['analytic_paired_gap']) for r in domain) < 1e-12
    assert all(r['mean_equality'] == 0 and r['reference']['minimum_entropy_production'] >= 0 for r in domain)
    weak = []
    for g, tau in [(0., .7), (.01, .7), (.5, .7), (.5, .03), (.5, .003), (.5, 0.)]:
        e = ref.experiment(1.2, .8, .2, tau)
        baseline = vector(source, e, 1.03)
        harmonic = (1-1j*g*e['frequency']*tau)*complex(*baseline[1:])
        expected = np.array([baseline[0], harmonic.real, harmonic.imag])
        row = dict(g=g, input=e, source=baseline.tolist(), oracle=expected.tolist(),
                   paired_gap=error(baseline[1:], expected[1:]), scope='Author-only limits; fixed task g=.5 and tau[.2,1.2].')
        if tau in [.7, .03]:
            row['reference'] = ref.first_law_reference(e, 1.03, g=g)
            row['reference_error'] = float(np.max(abs(expected-row['reference']['values'])))
            assert row['reference_error'] < 1e-8
        weak.append(row)
    REPORT['weak_limits'] = weak
    REPORT['signals'] = {k: dict(rms=float(np.sqrt(np.mean(v*v))),
        source_error=error(source.predict_at(groups[k], ref.TRUE_PARAMETER), v)) for k, v in truths.items()}
    REPORT['local_controls'] = local_controls()
    assert LOADED == {k: digest(BASE/k) for k in LOADED}, 'Source changed during validation.'
    REPORT.update(status='science_and_local_controls_complete', seconds=time.monotonic()-start)


if __name__ == '__main__':
    output = RESULTS/'hydrodynamic-heating-r4-validation.json'
    assert not output.exists(), 'Preserve previous author attempts before any rerun.'
    try:
        run()
    except Exception:
        REPORT.update(status='failed_author_validation', traceback=traceback.format_exc())
        raise
    finally:
        output.write_text(json.dumps(REPORT, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k: REPORT[k] for k in ['status', 'seconds', 'actual_data', 'local_controls'] if k in REPORT}, indent=2))
