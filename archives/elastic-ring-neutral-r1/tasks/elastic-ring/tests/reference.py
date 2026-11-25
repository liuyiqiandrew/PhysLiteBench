"""Cartesian spring-Hessian reference for the equilibrium preparation."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss

TRUE_PARAMETER = .9


def rhombus(angle):
    first = np.array([1., 0.])
    second = np.array([np.cos(angle), np.sin(angle)])
    positions = np.array([-first-second, first-second, first+second, -first+second])/2
    tangent = np.zeros((8, 2))
    tangent[:, 0] = np.column_stack((-positions[:, 1], positions[:, 0])).ravel()
    derivative = np.array([-np.sin(angle), np.cos(angle)])
    tangent[:, 1] = (np.array([-derivative, -derivative, derivative, derivative])/2).ravel()
    return positions, tangent


def transverse_weight(angle):
    positions, tangent = rhombus(angle)
    gradients = np.zeros((4, 8))
    for side in range(4):
        other = (side+1) % 4
        direction = positions[other]-positions[side]
        direction /= np.linalg.norm(direction)
        gradients[side, 2*side:2*side+2] = -direction
        gradients[side, 2*other:2*other+2] = direction
    hessian = gradients.T@gradients
    normal_eigenvalues = np.linalg.eigvalsh(hessian)[-4:]
    volume = np.sqrt(np.linalg.det(tangent.T@tangent))
    return volume/np.sqrt(np.prod(normal_eigenvalues))


@lru_cache(maxsize=64)
def shape_grid(cutoff, order=96):
    nodes, quadrature = leggauss(order)
    angle = cutoff+(nodes+1)*(np.pi-2*cutoff)/2
    density = np.array([transverse_weight(a) for a in angle])
    return angle, quadrature*density


def predict(experiments, stiffness=TRUE_PARAMETER, order=96):
    result = []
    for experiment in experiments:
        if experiment['readout'] == 'torque':
            result.append(-stiffness*np.sin(experiment['angle']-experiment['preferred']))
            continue
        angles, base = shape_grid(experiment['cutoff'], order)
        weight = base*np.exp(-stiffness*(1-np.cos(angles-experiment['preferred']))/experiment['temperature'])
        moment = np.sin(angles) if experiment['readout'] == 'sine' else np.cos(2*angles)
        result.append(float(weight@moment/weight.sum()))
    return np.array(result)


def calibration_inputs():
    return [dict(readout='torque', angle=float(angle), preferred=float(preferred))
            for _ in range(2) for preferred in np.linspace(.55, np.pi-.55, 6)
            for angle in np.linspace(.28, np.pi-.28, 12)]


def hidden_inputs():
    groups = {}
    for name, preferred in [('symmetric', np.pi/2), ('lower_preference', .65), ('upper_preference', 2.45)]:
        groups[name] = [dict(readout=readout, temperature=temperature, preferred=preferred, cutoff=cutoff)
                       for temperature, cutoff in [(.55, .23), (.8, .33), (1.1, .42), (1.4, .27), (1.7, .37), (1.95, .45)]
                       for readout in ['sine', 'cosine2']]
    return groups
