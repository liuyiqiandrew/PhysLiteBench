"""Route-marked characteristic generator for stationary regeneration work."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigvals

TRUE_PARAMETER = 1.07


@lru_cache(maxsize=4096)
def _unit_power(energies, attempts, affinities, step):
    rates = np.asarray(attempts).reshape(3, 2)
    generator = np.zeros((3, 3), dtype=complex)
    for edge in range(3):
        destination = (edge + 1) % 3
        for route in range(2):
            cost = affinities[route]
            forward = rates[edge, route] * np.exp((cost + energies[edge] - energies[destination])/2)
            reverse = rates[edge, route] * np.exp((-cost - energies[edge] + energies[destination])/2)
            generator[destination, edge] += forward * np.exp(1j*step*cost)
            generator[edge, destination] += reverse * np.exp(-1j*step*cost)
            generator[edge, edge] -= forward
            generator[destination, destination] -= reverse
    values = eigvals(generator)
    eigenvalue = values[np.argmin(np.abs(values))]
    return float(eigenvalue.imag/step)


def predict(experiments, scale, step=1e-5):
    output = []
    for e in experiments:
        args = (tuple(e['energies']), tuple(np.asarray(e['attempts']).ravel()), tuple(e['affinities']))
        fine = _unit_power(*args, step)
        coarse = _unit_power(*args, 2*step)
        output.append(scale*(4*fine-coarse)/3)
    return np.array(output)


def calibration_inputs():
    inputs = []
    for affinity in (.4, .9, 1.5):
        for energies in ((0., -.3, .2), (0., .3, -.2), (0., .1, .3)):
            for varied in (False, True):
                attempts = np.array([[.6, 1.2], [1.3, .7], [.8, 1.1]])
                if varied:
                    attempts *= np.array([1.15, .8, 1.1])[:, None]
                inputs.append({'energies':list(energies), 'attempts':attempts.tolist(),
                               'affinities':[affinity, affinity]})
    return inputs


def hidden_inputs():
    patterns = [
        ([0., -.3, .25], [[.65,1.3],[1.2,.7],[.8,1.15]]),
        ([0., .2, -.3], [[1.35,.6],[.75,1.25],[1.1,.9]]),
        ([0., .1, .3], [[.9,1.1],[1.4,.6],[.7,1.3]]),
        ([0., -.2, -.1], [[1.2,.8],[.65,1.35],[1.3,.7]]),
    ]
    groups = {}
    for name, a in [('high_drive',[2.3,.3]),('moderate_drive',[1.65,.6]),
                    ('barrier_variation',[1.9,.4]),('equal_drive_anchor',[1.1,1.1])]:
        groups[name] = [{'energies':e, 'attempts':v, 'affinities':a} for e,v in patterns]
    return groups
