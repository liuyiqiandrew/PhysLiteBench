"""Model for the force on the right wall of the one-particle cavity."""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar


@lru_cache(maxsize=4096)
def _spectrum(length, strength, mass, count=32):
    """Return ``(energies, wall_pressures)`` for the lowest states.

    The pressure is obtained from the Dirichlet-wall shape derivative,
    ``F_right = |psi'(right wall)|**2/(2*mass)``.  This is important for a
    nonzero delta strength: in that case the force is not simply
    ``2*energy/length``.
    """
    length = float(length)
    strength = float(strength)
    mass = float(mass)
    coefficient = 1.0 / (2.0 * mass)
    coupling = mass * strength * length / 2.0

    states = []
    for n in range(count):
        # For an even state, z=k*length/2 is in ((n+1/2)pi,(n+1)pi).
        lower = (n + 0.5) * np.pi

        def equation(delta):
            return (lower + delta) * np.tan(delta) - coupling

        delta = brentq(
            equation,
            0.0,
            np.pi / 2.0 - 1.0e-12,
            xtol=2.0e-14,
            rtol=1.0e-14,
        )
        z = lower + delta
        k = 2.0 * z / length
        energy = coefficient * k * k

        # On the right half, psi is proportional to sin(k*(length/2-x)).
        # The expression below is the normalization denominator over both
        # halves, so the boundary derivative gives the wall pressure.
        normalization_denominator = length - np.sin(2.0 * z) / k
        pressure = k * k / (mass * normalization_denominator)
        states.append((energy, pressure))

        # Odd states vanish at x=0 and are unaffected by the defect.
        z = (n + 1.0) * np.pi
        k = 2.0 * z / length
        energy = coefficient * k * k
        pressure = k * k / (mass * length)
        states.append((energy, pressure))

    states.sort(key=lambda state: state[0])
    energies = np.asarray([state[0] for state in states], dtype=float)
    pressures = np.asarray([state[1] for state in states], dtype=float)
    return energies, pressures


def levels(length, strength, mass, count=32):
    """Return the lowest one-particle energy levels."""
    return _spectrum(length, strength, mass, count)[0]


def force(experiment, mass):
    length = float(experiment['length'])
    strength = float(experiment['strength'])
    temperature = float(experiment['temperature'])
    energy, pressure = _spectrum(length, strength, mass)
    # Shift by the ground state to keep the Boltzmann weights well-scaled.
    weight = np.exp(-(energy - energy[0]) / temperature)
    weight /= weight.sum()
    return float(weight @ pressure)


def predict_at(experiments, mass):
    return np.asarray([force(experiment, mass) for experiment in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mass=None

    def fit(self, records):
        records = list(records)
        inputs = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record.get('sigma', 0.001) for record in records], dtype=float)
        if not inputs:
            raise ValueError('records must contain at least one calibration record')
        if not (np.all(np.isfinite(values)) and np.all(np.isfinite(sigma)) and np.all(sigma > 0)):
            raise ValueError('calibration values and uncertainties must be finite and positive')

        def objective(mass):
            residual = (predict_at(inputs, mass) - values) / sigma
            return float(residual @ residual)

        result = minimize_scalar(
            objective,
            bounds=(0.4, 0.9),
            method='bounded',
            options={'xatol': 1.0e-12, 'maxiter': 200},
        )
        candidates = [
            (result.fun, result.x),
            (objective(0.4), 0.4),
            (objective(0.9), 0.9),
        ]
        _, self.mass = min(candidates, key=lambda item: item[0])
        self.mass = float(self.mass)
        return self

    def predict(self, experiments):
        return predict_at(experiments,self.mass)
