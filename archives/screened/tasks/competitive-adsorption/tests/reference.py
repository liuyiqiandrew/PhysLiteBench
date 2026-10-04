"""Solve mass-action stationarity with one normalized site population."""
import numpy as np

PARAMETERS = ['affinity_a', 'affinity_b']
TRUE_PARAMETERS = [1.6, .85]
BOUNDS = (.2, 3.)


def fractions(concentration_a, concentration_b, affinities):
    wa = affinities[0]*concentration_a
    wb = affinities[1]*concentration_b
    equations = [[1., 1., 1.], [-wa, 1., 0.], [-wb, 0., 1.]]
    return np.linalg.solve(equations, [1., 0., 0.])


def predict(experiments, affinities):
    result = []
    for e in experiments:
        ca, cb = e['concentration_a'], e['concentration_b']
        index = 1 if e['species'] == 'A' else 2
        occupancy = fractions(ca, cb, affinities)[index]
        if e['protocol'] == 'displacement':
            initial = fractions(ca if index == 1 else 0., cb if index == 2 else 0., affinities)[index]
            result.append(initial-occupancy)
        else:
            result.append(occupancy)
    return np.array(result)


def calibration_inputs():
    result = []
    for c in np.linspace(.03, 1.15, 48):
        for species in ['A', 'B']:
            result.append(dict(protocol='equilibrium', species=species,
                               concentration_a=float(c) if species == 'A' else 0.,
                               concentration_b=float(c) if species == 'B' else 0.))
    return result


def hidden_inputs():
    result = {}
    for protocol in ['equilibrium', 'displacement']:
        for species in ['A', 'B']:
            result[protocol+'_'+species] = [dict(protocol=protocol, species=species,
                concentration_a=float(ca), concentration_b=float(cb))
                for ca in np.linspace(.20, .55, 8) for cb in np.linspace(.25, .75, 7)]
    return result
