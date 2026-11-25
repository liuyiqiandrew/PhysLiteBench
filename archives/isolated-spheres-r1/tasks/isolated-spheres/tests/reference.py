import itertools
import numpy as np

TRUE_PARAMETER = 1.03


def free_energy_force(separation, radius, isolated=True, terms=128):
    psi = np.arccosh(separation / (2 * radius))
    dpsi = 1 / np.sqrt(separation**2 - 4 * radius**2)
    n = np.arange(1, terms + 1, dtype=float)
    odd = 2 * n - 1
    z = np.exp(-2 * odd * psi)
    energy = .5 * np.sum(odd * np.log1p(-z))
    force = -np.sum(odd**2 * z / (1 - z)) * dpsi
    if isolated:
        def capacitance(indices):
            x = indices * psi
            inverse_sinh = 2 * np.exp(-x) / (1 - np.exp(-2*x))
            value = radius * np.sinh(psi) * np.sum(inverse_sinh)
            derivative = radius * (np.cosh(psi) * np.sum(inverse_sinh)
                - np.sinh(psi) * np.sum(indices * inverse_sinh / np.tanh(x))) * dpsi
            return value, derivative
        diagonal, ddiagonal = capacitance(odd)
        mutual, dmutual = capacitance(2*n)
        determinant = diagonal**2 - mutual**2
        energy += .5 * np.log(determinant / radius**2)
        force -= (diagonal*ddiagonal - mutual*dmutual) / determinant
    return float(energy), float(force)


def predict(experiments, radius=TRUE_PARAMETER):
    return np.array([radius**3*e['field'] if e['kind']=='dipole'
        else free_energy_force(e['separation'],radius)[1] for e in experiments])


def calibration_inputs():
    return [dict(kind='dipole',field=f) for f,repeat in itertools.product(
        [-1.,-.8,-.6,-.4,-.2,.2,.4,.6,.8,1.],range(24))]


def hidden_inputs():
    return {
        'near_force':[dict(kind='force',separation=x) for x in [3.,3.1,3.2,3.3]],
        'middle_force':[dict(kind='force',separation=x) for x in [3.4,3.5,3.6,3.7]],
        'far_force':[dict(kind='force',separation=x) for x in [3.9,4.1,4.3,4.5]],
        'dipole_anchors':[dict(kind='dipole',field=x) for x in [-.93,-.31,.17,.67]]
    }
