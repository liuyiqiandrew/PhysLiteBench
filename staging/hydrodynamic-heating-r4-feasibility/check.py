"""Bounded thermoactive-contact check; no harness or model evaluation."""
import hashlib
import json
from itertools import product
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.integrate import solve_ivp

import r3_oracle as source
import r3_reference as spatial

BASE = Path(__file__).resolve().parent
G = 0.5
REPORT = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entropy_prediction(e, p, g=G):
    """Modal mechanics plus the elastic entropy inferred from force(T)."""
    controls = {k: e[k] for k in ('frequency', 'thickness', 'friction', 'relaxation')}
    mean, dashpot = source.contact_response(**controls, plasma_frequency=p)
    # At fixed T0, T0*s_dot = -g*k0*e*e_dot. In normalized units
    # the entropy contribution to the 2w amplitude is -i*g*w*tau*Qd2.
    harmonic = dashpot - 1j*g*e['frequency']*e['relaxation']*dashpot
    return np.array([mean, harmonic.real, harmonic.imag])


def dashpot_prediction(e, p):
    return entropy_prediction(e, p, 0.0)


def first_law_reference(e, p, g=G, field_tol=1e-9, time_tol=2e-9,
                        settle_periods=8, steps_per_period=64, periods=1):
    """Collocation fields, real extension ODE, and mechanical input minus Udot.

    Choose rho=1 as a harmless normalization: v=J/p, kappa=alpha,
    k0=alpha/tau. E0=1 so division by incident mean power multiplies by2.
    No complex heat correction is used in this calculation.
    """
    w, alpha, tau = (e[k] for k in ('frequency', 'friction', 'relaxation'))
    assert tau > 0 and w > 0
    fields = spatial.bvp_fields(e, p, field_tol)
    j = fields([0.0])[2, 0]
    period = 2*np.pi/w
    k0 = alpha/tau

    def slip(t):
        return float(np.real(j*np.exp(-1j*w*t)))/p

    def extension_rate(t, x):
        return slip(t) - x/tau

    options = dict(method='DOP853', rtol=time_tol, atol=time_tol*0.01,
                   max_step=period/steps_per_period)
    settled = solve_ivp(lambda t, y: [extension_rate(t, y[0])],
                        (0.0, settle_periods*period), [0.0], **options)
    assert settled.success, settled.message
    initial = settled.y[0, -1]

    def rhs(t, y):
        x = y[0]
        v = slip(t)
        xd = extension_rate(t, x)
        traction = k0*x
        work = 2*traction*v
        # U=(k-T*k')*e^2/2=(1-g)*k0*e^2/2 at the clamped T0.
        udot = 2*(1-g)*k0*x*xd
        heat = work-udot
        dashpot = 2*alpha*(x/tau)**2
        cosine, sine = np.cos(2*w*t), np.sin(2*w*t)
        return [xd, heat, heat*cosine, heat*sine,
                dashpot, dashpot*cosine, dashpot*sine, work, udot]

    end = periods*period
    measured = solve_ivp(rhs, (0.0, end), [initial, 0, 0, 0, 0, 0, 0, 0, 0],
                         dense_output=True, **options)
    assert measured.success, measured.message
    final = measured.y[:, -1]
    values = np.array([final[1], 2*final[2], 2*final[3]])/end
    dashpot = np.array([final[4], 2*final[5], 2*final[6]])/end
    ts = np.linspace(0, period, 257)
    x = measured.sol(ts)[0]
    v = np.real(j*np.exp(-1j*w*ts))/p
    xd = v-x/tau
    work = 2*k0*x*v
    udot = 2*(1-g)*k0*x*xd
    heat = work-udot
    entropy_heat = 2*alpha*(x/tau)**2+2*g*k0*x*xd
    integrated_udot = final[8]
    endpoint_energy = (1-g)*k0*(final[0]**2-initial**2)
    modal_j = source.modal_fields(e, p)([0.0])[2, 0]
    return dict(values=values.tolist(), dashpot=dashpot.tolist(),
                field_error=float(abs(j-modal_j)),
                closure_error=float(np.max(abs(heat-entropy_heat))),
                energy_integral_error=float(abs(integrated_udot-endpoint_energy)),
                periodicity=float(abs(final[0]-initial)),
                mean_storage_rate=float(final[8]/end),
                heat_waveform_min=float(heat.min()), heat_waveform_max=float(heat.max()),
                minimum_entropy_production=float(np.min(alpha*(x/tau)**2)),
                mean_work=float(final[7]/end), settle_periods=settle_periods,
                measurement_periods=periods, time_tol=time_tol,
                field_tol=field_tol, max_step=period/steps_per_period,
                ode_evaluations=settled.nfev+measured.nfev)


def nrms(actual, truth):
    return float(np.linalg.norm(np.asarray(actual)-truth)/np.linalg.norm(truth))


def group_prediction(experiments, p, physical=True):
    out = []
    for e in experiments:
        vector = entropy_prediction(e, p) if physical else dashpot_prediction(e, p)
        out.append(vector[{'mean': 0, 'in_phase': 1, 'quadrature': 2}[e['readout']]])
    return np.array(out)


def run():
    start = time.monotonic()
    loaded = {p.name: digest(p) for p in BASE.glob('*') if p.is_file() and p.suffix in ['.py', '.json']}
    REPORT.update(status='in_progress', fixed_g=G, loaded_inputs_sha256=loaded,
                  new_model_runs=0, new_docker_runs=0)
    domain = []
    controls = [(spatial.experiment(w, d, a, t), p) for p, w, d, a, t in
                product([.85, 1.15], [.7, 1.5], [.3, 1.2], [.08, .4], [.2, 1.2])]
    rng = np.random.default_rng(194071)
    controls += [(spatial.experiment(rng.uniform(.7, 1.5), rng.uniform(.3, 1.2),
                                    rng.uniform(.08, .4), rng.uniform(.2, 1.2)),
                  float(rng.uniform(.85, 1.15))) for _ in range(16)]
    for e, p in controls:
        y, wrong = entropy_prediction(e, p), dashpot_prediction(e, p)
        ref = first_law_reference(e, p)
        domain.append(dict(input=e, plasma_frequency=p, oracle=y.tolist(), source=wrong.tolist(),
                           reference=ref, reference_error=float(max(abs(y-ref['values']))),
                           paired_gap=nrms(wrong[1:], y[1:]),
                           paired_signal_rms=float(np.linalg.norm(y[1:])/np.sqrt(2)),
                           analytic_gap=G*e['frequency']*e['relaxation']/np.sqrt(1+(G*e['frequency']*e['relaxation'])**2)))
    REPORT['domain'] = dict(corners=32, interiors=16, rows=domain)

    # Separate each refinement: field tolerance, time discretization, settling,
    # and number of measured periods. Never conflate these with noise.
    refinements = []
    for index in [0, 7, 16, 31, 33, 41]:
        e, p = controls[index]
        base = domain[index]['reference']
        variants = {}
        for label, kwargs in [
                ('field', dict(field_tol=1e-10)),
                ('time', dict(time_tol=2e-10, steps_per_period=128)),
                ('settling', dict(settle_periods=12)),
                ('measurement_periods', dict(periods=2))]:
            result = first_law_reference(e, p, **kwargs)
            variants[label] = dict(result=result, maximum_change=float(max(abs(np.array(result['values'])-base['values']))))
        refinements.append(dict(domain_index=index, variants=variants))
    REPORT['refinement'] = refinements

    # Retain all r3 paired diagnostics and mean anchors without tuning controls.
    groups = spatial.hidden_inputs()
    reference_cache = {}
    truths = {}
    scored = []
    for group, experiments in groups.items():
        values = []
        for e in experiments:
            key = tuple(e[k] for k in ('frequency', 'thickness', 'friction', 'relaxation'))
            if key not in reference_cache:
                reference_cache[key] = first_law_reference(e, spatial.TRUE_PARAMETER)
            ref = reference_cache[key]
            i = {'mean': 0, 'in_phase': 1, 'quadrature': 2}[e['readout']]
            values.append(ref['values'][i])
        truth = np.array(values)
        truths[group] = truth
        scored.append(dict(group=group, inputs=experiments, reference=truth.tolist(),
                           source=group_prediction(experiments, spatial.TRUE_PARAMETER, False).tolist(),
                           oracle=group_prediction(experiments, spatial.TRUE_PARAMETER).tolist(),
                           signal_rms=float(np.sqrt(np.mean(truth**2))),
                           source_error=nrms(group_prediction(experiments, spatial.TRUE_PARAMETER, False), truth),
                           oracle_error=nrms(group_prediction(experiments, spatial.TRUE_PARAMETER), truth)))
    REPORT['inherited_groups'] = scored

    rows = json.loads((BASE/'r3_calibration.json').read_text())
    fitted = source.Model().fit(rows).plasma_frequency
    inputs = spatial.calibration_inputs()
    cal_bias = []
    for e in inputs:
        ref = first_law_reference(e, spatial.TRUE_PARAMETER)
        value = entropy_prediction(e, spatial.TRUE_PARAMETER)[0]
        cal_bias.append(dict(input=e, first_law_mean=ref['values'][0], exact_mean=float(value),
                             absolute_error=abs(ref['values'][0]-value), fixed_sigma_units=abs(ref['values'][0]-value)/.0002))
    old = json.loads((BASE/'r3_validation.json').read_text())
    transferred = []
    for r in old['noise']['rows']:
        p = r['oracle']['parameter']
        transferred.append(dict(seed_index=r['seed_index'], fitted_parameter=p,
            calibration_chi2=r['oracle']['calibration_chi2'],
            relative_parameter_error=r['oracle']['relative_parameter_error'],
            oracle={name: nrms(group_prediction(es, p), truths[name]) for name, es in groups.items()},
            source={name: nrms(group_prediction(es, p, False), truths[name]) for name, es in groups.items()}))
    REPORT['calibration'] = dict(records=len(rows), fixed_sigma=.0002, actual_fit=fitted,
        actual_fit_difference_from_r3=float(abs(fitted-old['actual_data']['oracle']['parameter'])),
        mean_reference=cal_bias,
        exact_transfer='The complete r4 mean map and fit objective are exactly the r3 oracle map for every p. Its41 noiseless recovery/profile checks and256 noisy oracle fitted parameters therefore transfer; the frozen rows are reused, not called fresh draws. New readouts below are rescored against the independent first-law reference.',
        old_identifiability_summary=dict(noiseless_fits=len(old['identifiability']['noiseless_fits']),
            maximum_fit_error=max(r['max_error'] for r in old['identifiability']['noiseless_fits']),
            minimum_sampled_secant_derivative=old['identifiability']['minimum_secant_derivative']),
        transferred_noise_rows=transferred)

    limits = []
    for g, tau in [(0., .7), (.01, .7), (G, .7), (G, .03), (G, .003), (G, 0.)]:
        e = spatial.experiment(1.2, .8, .2, tau)
        y, wrong = entropy_prediction(e, 1.03, g), dashpot_prediction(e, 1.03)
        row = dict(g=g, input=e, oracle=y.tolist(), source=wrong.tolist(), paired_gap=nrms(wrong[1:], y[1:]),
                   scope='Author-only weak-material or small-memory limit; fixed task candidate g=.5 and tau[.2,1.2].')
        if tau in [.7, .03]:
            row['reference'] = first_law_reference(e, 1.03, g)
            row['reference_error'] = float(max(abs(y-row['reference']['values'])))
        limits.append(row)
    REPORT['weak_limits'] = limits
    summary = dict(
        maximum_domain_reference_error=max(r['reference_error'] for r in domain),
        maximum_field_error=max(r['reference']['field_error'] for r in domain),
        maximum_first_law_entropy_difference=max(r['reference']['closure_error'] for r in domain),
        maximum_energy_integral_error=max(r['reference']['energy_integral_error'] for r in domain),
        maximum_periodicity_error=max(r['reference']['periodicity'] for r in domain),
        maximum_reference_refinement=max(v['maximum_change'] for r in refinements for v in r['variants'].values()),
        minimum_sampled_pair_gap=min(r['paired_gap'] for r in domain),
        minimum_sampled_signal_rms=min(r['paired_signal_rms'] for r in domain),
        minimum_whole_contact_instantaneous_heat=min(r['reference']['heat_waveform_min'] for r in domain),
        maximum_noise_oracle_error=max(max(r['oracle'].values()) for r in transferred),
        minimum_noise_source_diagnostic_error=min(r['source'][k] for r in transferred for k in groups if k!='mean_anchor'),
        maximum_calibration_reference_bias_sigma=max(r['fixed_sigma_units'] for r in cal_bias))
    REPORT['summary'] = summary
    assert summary['maximum_domain_reference_error'] < 1e-7
    assert summary['maximum_reference_refinement'] < 1e-7
    assert summary['maximum_first_law_entropy_difference'] < 1e-12
    assert summary['maximum_energy_integral_error'] < 1e-8
    assert summary['maximum_noise_oracle_error'] < .04
    assert summary['minimum_noise_source_diagnostic_error'] > .04
    assert max(r['source']['mean_anchor'] for r in transferred) < .04
    assert all(r['calibration_chi2'] < 1.5 and r['relative_parameter_error'] < .03 for r in transferred)
    assert len(transferred) == 256 and all(r['sigma'] == .0002 for r in rows)
    assert all(abs(r['paired_gap']-r['analytic_gap']) < 1e-12 for r in domain)
    assert all(digest(BASE/name) == value for name, value in loaded.items())
    REPORT.update(status='bounded_science_complete', seconds=time.monotonic()-start)


if __name__ == '__main__':
    output = BASE/'report.json'
    assert not output.exists(), 'Preserve every attempt; select a new report path before another invocation.'
    try:
        run()
    except Exception:
        REPORT.update(status='failed_bounded_check', traceback=traceback.format_exc())
        raise
    finally:
        output.write_text(json.dumps(REPORT, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k: REPORT[k] for k in ['status', 'seconds', 'summary'] if k in REPORT}, indent=2))
