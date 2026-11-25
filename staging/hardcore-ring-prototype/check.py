"""Unevaluated alternate hard-core geometry: exchange across a closing bond."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigh

LENGTH = 8
OCCUPIED = {2: (3, 4), 3: (2, 3, 4), 4: (2, 3, 4, 5)}


def potential(amplitude, asymmetry):
    angle = 2 * np.pi * np.arange(LENGTH) / LENGTH
    return amplitude * np.cos(angle) + asymmetry * np.sin(angle)


def one_body(hopping, number, flux, amplitude, asymmetry, correct=True):
    matrix = np.diag(potential(amplitude, asymmetry)).astype(complex)
    for j in range(LENGTH):
        k = (j + 1) % LENGTH
        bond = -hopping * np.exp(1j * flux / LENGTH)
        if correct and j == LENGTH - 1:
            bond *= (-1) ** (number - 1)
        matrix[j, k] = bond
        matrix[k, j] = bond.conjugate()
    return matrix


def density(hopping, number, flux, amplitude, asymmetry, duration, correct=True):
    energy, vectors = eigh(one_body(hopping, number, flux, amplitude, asymmetry, correct))
    propagator = (vectors * np.exp(-1j * energy * duration)) @ vectors.conj().T
    return np.sum(abs(propagator[:, OCCUPIED[number]]) ** 2, axis=1)


def fock_density(hopping, number, flux, amplitude, asymmetry, duration):
    states = [sum(1 << j for j in sites) for sites in combinations(range(LENGTH), number)]
    lookup = {bits: index for index, bits in enumerate(states)}
    occupations = np.array([[(bits >> j) & 1 for j in range(LENGTH)] for bits in states])
    matrix = np.diag(occupations @ potential(amplitude, asymmetry)).astype(complex)
    for column, bits in enumerate(states):
        for j in range(LENGTH):
            k = (j + 1) % LENGTH
            for target, origin, phase in [(j, k, 1), (k, j, -1)]:
                if (bits >> origin) & 1 and not (bits >> target) & 1:
                    moved = bits ^ (1 << origin) ^ (1 << target)
                    matrix[lookup[moved], column] += -hopping * np.exp(1j * phase * flux / LENGTH)
    energies, vectors = eigh(matrix)
    initial = lookup[sum(1 << j for j in OCCUPIED[number])]
    state = (vectors * np.exp(-1j * energies * duration)) @ vectors[initial].conj()
    return abs(state) ** 2 @ occupations


def main():
    rng = np.random.default_rng(640831)
    independent = []
    for number in [2, 3, 4]:
        for _ in range(20):
            controls = [rng.uniform(.8, 1.2), number, rng.uniform(-np.pi, np.pi),
                        rng.uniform(0, .7), rng.uniform(-.4, .4), rng.uniform(0, 4)]
            a = density(*controls)
            b = fock_density(*controls)
            independent.append({'controls': controls, 'maximum_error': float(abs(a-b).max()),
                                'number_error': float(abs(a.sum()-number))})
    groups = []
    for number in [2, 4]:
        for hopping in [.8, 1., 1.2]:
            for flux in [0., 1., -2.]:
                exact, shortcut = [], []
                for amplitude, asymmetry in [(0., 0.), (.4, -.3), (.7, .4)]:
                    for duration in [1.5, 2., 2.5, 3., 3.5]:
                        args = hopping, number, flux, amplitude, asymmetry, duration
                        exact.extend(density(*args))
                        shortcut.extend(density(*args, correct=False))
                exact, shortcut = np.array(exact), np.array(shortcut)
                groups.append({'number': number, 'hopping': hopping, 'flux': flux,
                               'relative_rms': float(np.linalg.norm(exact-shortcut)/np.linalg.norm(exact))})
    calibration = []
    for hopping in np.linspace(.8, 1.2,41):
        for flux in [0., 1., -2.]:
            for duration in [.12, .2, .3, .4]:
                args = 3, flux, .4, -.3, duration
                derivative = (density(hopping+1e-5, *args)-density(hopping-1e-5, *args))/2e-5
                calibration.append(float(min(derivative[[1, 5]])))
    report = {'status': 'prototype_only_no_task_or_model_trials',
              'family': 'Alternate ring revision of hard-core boson dynamics; not an independent new family.',
              'independent_checks': independent, 'groups': groups,
              'minimum_sampled_calibration_derivative': min(calibration),
              'calibration_equivalence': 'Exact for N=3: (-1)^(N-1)=1.',
              'maximum_independent_error': max(x['maximum_error'] for x in independent),
              'minimum_group_gap': min(x['relative_rms'] for x in groups)}
    assert report['maximum_independent_error'] < 1e-11
    Path(__file__).with_name('assessment.json').write_text(json.dumps(report, indent=2)+'\n')
    print({k: report[k] for k in ['maximum_independent_error', 'minimum_group_gap', 'minimum_sampled_calibration_derivative']})


if __name__ == '__main__':
    main()
