"""Independent cell-entropy constitutive elimination and heat-flux balance."""
from functools import lru_cache

import numpy as np
from scipy.linalg import eig, solve

PARAMETER = 'conductivity'
TRUE_PARAMETER = 145.
LENGTH, AREA, T0, CE, CB = .05, 1e-4, 300., 1e6, 6.
E0 = E1 = 2e9
SCALE = .002


def calibration_inputs():
    return [dict(mean=0., first=amplitude, second=0., bath_initial=0.,
                 chi=0., contact=0., time=time, observable='first')
            for amplitude in [-.8, .4, .8] for time in [.2, 1., 4., 12., 40.]]


def hidden_inputs():
    result = {}
    for name, contact, coefficients in [
            ('insulated', 0., (.2, 0., .8, -.6)),
            ('contact_0.4', .4, (-.3, .3, .4, .8)),
            ('contact_1.1', 1.1, (-.3, .3, .4, .8))]:
        result[name] = [dict(zip(('mean', 'first', 'second', 'bath_initial'), coefficients),
                            chi=1., contact=contact, time=time, observable='mean')
                        for time in [1., 5., 20., 80.]]
    return result


def generator_and_maps(conductivity, contact, chi, cells):
    dx = LENGTH/cells
    x = (np.arange(cells)+.5)*dx
    beta0 = E0*.002*(1+.65*np.cos(2*np.pi*x/LENGTH))
    beta1 = chi*beta0
    total = beta0+beta1
    eta = E1*12*(1+.97*np.cos(2*np.pi*x/LENGTH))
    basis = np.eye(2*cells+1)
    entropy = basis[:cells]
    z = SCALE*basis[cells:2*cells]
    body = basis[-1]
    denominator = E0+E1+T0/CE*total**2
    loading = (E1+T0/CE*total*beta1)[:, None]*z + total[:, None]*entropy
    stress = -np.sum(loading/denominator[:, None], axis=0)/np.sum(1/denominator)
    strain = (loading+stress)/denominator[:, None]
    theta = entropy-T0/CE*total[:, None]*strain+T0/CE*beta1[:, None]*z
    heat_rate = np.zeros((cells, 2*cells+1))
    # Positive flux is heat moving toward increasing cell index.
    flux = conductivity*AREA/dx*(theta[:-1]-theta[1:])
    heat_rate[:-1] -= flux
    heat_rate[1:] += flux
    conductance = 0. if contact == 0 else 1/(1/contact+dx/(2*conductivity*AREA))
    entering = conductance*(body-theta[0])
    heat_rate[0] += entering
    generator = np.empty_like(basis)
    generator[:cells] = heat_rate/(CE*AREA*dx)
    generator[cells:2*cells] = (E1*(strain-z)-beta1[:, None]*theta)/eta[:, None]/SCALE
    generator[-1] = -entering/CB
    profiles = np.column_stack([np.ones(cells), np.cos(np.pi*x/LENGTH), np.cos(2*np.pi*x/LENGTH), np.zeros(cells)])
    initial_strain = (beta0[:, None]*profiles-np.mean(beta0[:, None]*profiles, axis=0))/E0
    initial_z = initial_strain-beta1[:, None]*profiles/E1
    initial_entropy = profiles+T0/CE*(total[:, None]*initial_strain-beta1[:, None]*initial_z)
    initial = np.vstack([initial_entropy, initial_z/SCALE, [0.,0.,0.,1.]])
    observations = np.vstack([np.mean(theta, axis=0),
                              np.cos(np.pi*x/LENGTH)@theta/cells,
                              np.cos(2*np.pi*x/LENGTH)@theta/cells,body])
    return generator, initial, observations


@lru_cache(maxsize=192)
def decomposition(conductivity, contact, chi, cells):
    generator, initial, output = generator_and_maps(conductivity, contact, chi, cells)
    values, vectors = eig(generator)
    amplitudes = solve(vectors, initial)
    coefficients = (output@vectors)[:, :, None]*amplitudes[None, :, :]
    return values, coefficients


def predict(experiments, conductivity, cells=160):
    names = {'mean':0, 'first':1, 'second':2, 'bath':3}
    result = []
    for e in experiments:
        values, coefficients = decomposition(float(conductivity), float(e['contact']), float(e['chi']), cells)
        initial = np.array([e[k] for k in ('mean','first','second','bath_initial')])
        result.append(float((np.exp(values*e['time'])@coefficients[names[e['observable']]]@initial).real))
    return np.asarray(result)
