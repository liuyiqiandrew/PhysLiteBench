"""Bounded author prototype; no task package or model evaluation.

Two bands each contain M spinless one-particle states, at -W/2 and W/2.
All extensive observables are divided by 2M; kB=1. The material is closed
to particle exchange while a weak thermal bath equilibrates both bands.
"""
from itertools import product
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, gammaln, logsumexp


def state(width, temperature, filling, origin=0.):
    energies = origin + np.array([-.5, .5]) * width
    chemical = brentq(
        lambda mu: expit((mu - energies) / temperature).mean() - filling,
        energies[0] - 50 * temperature, energies[1] + 50 * temperature,
        xtol=5e-15)
    occupations = expit((chemical - energies) / temperature)
    return energies, chemical, occupations


def responses(width, temperature, filling, origin=0.):
    energies, chemical, occupations = state(width, temperature, filling, origin)
    fluctuation = occupations * (1 - occupations)
    relative = energies - chemical
    m0 = fluctuation.mean()
    m1 = (relative * fluctuation).mean()
    m2 = (relative ** 2 * fluctuation).mean()
    source = m2 / temperature ** 2
    physical = (m2 - m1 ** 2 / m0) / temperature ** 2
    return float(physical), float(source)


def energy(width, temperature, filling):
    energies, _, occupations = state(width, temperature, filling)
    return float(energies @ occupations / 2)


def energy_derivative(width, temperature, filling, fraction=1e-3):
    h = fraction * temperature
    v = [energy(width, temperature + j * h, filling) for j in [-2, -1, 1, 2]]
    return (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h)


def canonical(width, temperature, filling, band_size):
    """Exact fixed-N partition sum over upper-band occupation, independent
    of Fermi occupations, chemical potential and susceptibility subtraction.

    Integer particle number is held fixed for all temperatures at this M.
    The returned actual filling records finite-size rounding explicitly.
    """
    m = int(band_size)
    number = round(2 * m * filling)
    upper = np.arange(max(0, number - m), min(m, number) + 1, dtype=float)
    lower = number - upper
    log_weight = (2 * gammaln(m + 1) - gammaln(upper + 1)
                  - gammaln(m - upper + 1) - gammaln(lower + 1)
                  - gammaln(m - lower + 1) - width * upper / temperature)
    probability = np.exp(log_weight - logsumexp(log_weight))
    mean = float(probability @ upper)
    variance = float(probability @ (upper - mean) ** 2)
    mean_energy = width * (mean - number / 2) / (2 * m)
    heat_capacity = width ** 2 * variance / (2 * m * temperature ** 2)
    return float(mean_energy), float(heat_capacity), number / (2 * m)


def canonical_energy_derivative(width, temperature, filling, band_size, fraction=1e-3):
    h = fraction * temperature
    values = [canonical(width, temperature + j*h, filling, band_size)[0]
              for j in [-2, -1, 1, 2]]
    return (values[0] - 8*values[1] + 8*values[2] - values[3]) / (12*h)


def run():
    rng = np.random.default_rng(771504)
    report = {
        'status': 'bounded_prototype_only', 'model_evaluations': 0,
        'task_packages': 0,
        'domain': {'width': [.8, 1.2], 'temperature': [.1, .6], 'filling': [.1, .9]},
        'units': 'energy E0, temperature E0/kB, heat capacity kB per available one-particle state',
    }
    domain = list(product([.8, 1.2], [.1, .6], [.1, .5, .9]))
    domain += [tuple(rng.uniform([.8, .1, .1], [1.2, .6, .9])) for _ in range(48)]
    max_energy_error = max_derivative_refinement = max_origin = max_particle_hole = 0.
    min_physical = min_source = float('inf')
    canonical_rows = []
    for width, temperature, filling in domain:
        physical, source = responses(width, temperature, filling)
        derivative = energy_derivative(width, temperature, filling)
        refined = energy_derivative(width, temperature, filling, 5e-4)
        max_energy_error = max(max_energy_error, abs(refined - physical))
        max_derivative_refinement = max(max_derivative_refinement, abs(refined - derivative))
        shifted = responses(width, temperature, filling, 2.3)
        reflected = responses(width, temperature, 1 - filling)
        max_origin = max(max_origin, np.max(np.abs(np.array(shifted) - [physical, source])))
        max_particle_hole = max(max_particle_hole, np.max(np.abs(np.array(reflected) - [physical, source])))
        min_physical = min(min_physical, physical)
        min_source = min(min_source, source)
        # Subtract the thermodynamic prediction at each exact finite filling.
        # This isolates finite-size bias from integer-N rounding.
        errors = []
        for m in [512, 1024, 2048, 4096]:
            _, c, actual_filling = canonical(width, temperature, filling, m)
            limit = responses(width, temperature, actual_filling)[0]
            errors.append(c - limit)
        canonical_rows.append({'width': width, 'temperature': temperature, 'filling': filling,
                               'sizes_per_band': [512, 1024, 2048, 4096],
                               'signed_absolute_errors': errors})
    report['declared_domain'] = {
        'cases': len(domain), 'max_fixed_filling_energy_derivative_error': max_energy_error,
        'max_energy_derivative_refinement': max_derivative_refinement,
        'min_physical_heat_capacity': min_physical, 'min_source_heat_capacity': min_source,
        'max_energy_origin_shift_error': float(max_origin),
        'max_particle_hole_symmetry_error': float(max_particle_hole),
        'canonical_finite_size_checks': canonical_rows,
        'max_canonical_size4096_error': max(abs(r['signed_absolute_errors'][-1]) for r in canonical_rows),
    }
    # Exact rational fillings avoid all integer rounding in the extrapolation.
    rational_cases = list(product([.8, 1.2], [.1, .25, .6], [.1, .3, .5, .7, .9]))
    max_partition = max_partition_refinement = max_partition_derivative = 0.
    finite_errors = np.zeros(4)
    for width, temperature, filling in rational_cases:
        exact = responses(width, temperature, filling)[0]
        caps = np.array([canonical(width, temperature, filling, m)[1]
                         for m in [1000, 2000, 4000, 8000]])
        finite_errors = np.maximum(finite_errors, np.abs(caps - exact))
        coarse = (caps[0] - 6*caps[1] + 8*caps[2])/3
        fine = (caps[1] - 6*caps[2] + 8*caps[3])/3
        max_partition = max(max_partition, abs(fine - exact))
        max_partition_refinement = max(max_partition_refinement, abs(fine - coarse))
        direct = canonical_energy_derivative(width, temperature, filling, 1000)
        max_partition_derivative = max(max_partition_derivative, abs(direct - caps[0]))
    report['independent_canonical_partition'] = {
        'cases': len(rational_cases), 'sizes_per_band': [1000, 2000, 4000, 8000],
        'max_finite_size_errors': finite_errors.tolist(),
        'extrapolation': 'quadratic in inverse number of states per band, at exactly fixed rational filling',
        'max_extrapolated_absolute_error': max_partition,
        'max_extrapolation_refinement': max_partition_refinement,
        'max_direct_temperature_derivative_error': max_partition_derivative,
    }
    cal_temperatures = np.linspace(.1, .15, 9)
    max_cal_difference = max_cal_formula = max_fit_error = 0.
    derivative_values = []
    for width in np.linspace(.8, 1.2, 41):
        data = np.array([responses(width, t, .5)[0] for t in cal_temperatures])
        source = np.array([responses(width, t, .5)[1] for t in cal_temperatures])
        x = width/(4*cal_temperatures)
        formula = x*x/np.cosh(x)**2
        slope = 2*formula/width*(1 - x*np.tanh(x))
        derivative_values.extend(slope)
        max_cal_difference = max(max_cal_difference, np.max(abs(data-source)))
        max_cal_formula = max(max_cal_formula, np.max(abs(data-formula)))
        fit = minimize_scalar(lambda w: np.sum((np.array([responses(w,t,.5)[1]
                                   for t in cal_temperatures])-data)**2),
                              bounds=(.8,1.2),method='bounded',options={'xatol':1e-13})
        # Compare explicit endpoint objectives too, since the true width may be an endpoint.
        candidates = [float(fit.x), .8, 1.2]
        fitted = min(candidates, key=lambda w: np.sum((np.array([responses(w,t,.5)[1]
                        for t in cal_temperatures])-data)**2))
        max_fit_error = max(max_fit_error, abs(fitted-width))
    report['calibration'] = {
        'settings': len(cal_temperatures), 'temperatures': cal_temperatures.tolist(), 'filling': .5,
        'model_difference': float(max_cal_difference), 'analytic_formula_error': float(max_cal_formula),
        'noiseless_width_checks':41, 'max_width_recovery_error':max_fit_error,
        'maximum_derivative': float(max(derivative_values)), 'minimum_derivative':float(min(derivative_values)),
        'analytic_global_identifiability': 'x>=4/3 and x*tanh(x)>1; dC/dW=2*C/W*(1-x*tanh(x))<0 throughout calibration domain. Every calibration curve is injective.',
    }
    diagnostic = list(product(np.linspace(.8,1.2,9), np.linspace(.25,.55,7), [.1,.2,.3,.7,.8,.9]))
    gaps = []; signals = []
    for width, temperature, filling in diagnostic:
        physical, source = responses(width, temperature, filling)
        gaps.append((source-physical)/physical); signals.append(physical)
    report['doped_diagnostic_screen'] = {
        'cases':len(diagnostic),'width':[.8,1.2],'temperature':[.25,.55],
        'fillings':[.1,.2,.3,.7,.8,.9], 'min_relative_gap':min(gaps), 'max_relative_gap':max(gaps),
        'min_physical_heat_capacity':min(signals),'max_physical_heat_capacity':max(signals),
        'qualification':'Grid evidence, not a continuum bound or a task grading design.'}
    example_width = np.log(9.)
    example = responses(example_width, 1., .3)
    report['analytic_example'] = {'width_over_temperature':example_width,'filling':.3,
        'physical':example[0],'source':example[1],
        'formula_errors':[abs(example[0]-9*np.log(9.)**2/272),abs(example[1]-9*np.log(9.)**2/200)]}
    limits = []
    for filling in [.1,.3,.5,.7,.9]:
        for temperature in [20., 40., 80.]:
            physical, source = responses(1.,temperature,filling)
            limits.append({'filling':filling,'temperature':temperature,'physical':physical,'source':source,
              'scaled_physical':physical*temperature**2,
              'high_temperature_physical_coefficient':filling*(1-filling)/4,
              'source_high_temperature_limit':filling*(1-filling)*np.log(filling/(1-filling))**2})
    report['high_temperature_limits_outside_proposed_domain'] = limits
    report['all_checks_passed'] = bool(max_energy_error<1e-8 and max_partition<1e-7
         and max_fit_error<1e-7 and min(gaps)>.04 and min(signals)>.001
         and max_partition_derivative<1e-7)
    report['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['declared_domain','high_temperature_limits_outside_proposed_domain']},indent=2))


if __name__ == '__main__':
    run()
