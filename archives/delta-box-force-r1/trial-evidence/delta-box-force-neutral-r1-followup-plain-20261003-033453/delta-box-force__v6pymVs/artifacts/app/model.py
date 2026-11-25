"""Model for the force on the right wall of the one-particle cavity.

The spectrum is split into the odd states, which have a node at the defect,
and the even states, which are shifted by the delta potential.  The force is
computed from the probability-current stress at the wall (equivalently, from
the derivative of the energy with respect to the cavity length while keeping
the defect strength fixed).
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar


_MASS_MIN = 0.4
_MASS_MAX = 0.9


@lru_cache(maxsize=4096)
def _spectrum(length, strength, mass, count=24):
    """Return ``(energies, wall_forces)`` for the lowest states.

    ``count`` is the number of even and the number of odd states included.
    Returning the force of each level separately is important for a nonzero
    defect: the even-state boundary pressure is not simply ``2E/length``.
    """
    length = float(length)
    strength = float(strength)
    mass = float(mass)

    if length <= 0.0 or mass <= 0.0 or strength < 0.0:
        raise ValueError("length and mass must be positive; strength nonnegative")

    # q = k*length/2 and a = mass*strength*length/2.  The even-state
    # matching condition is q*cot(q) = -a.  Writing
    # q=(n+1/2)pi+d makes the root lie in d in [0, pi/2).
    a = mass * strength * length / 2.0
    energies = []
    wall_forces = []

    for n in range(int(count)):
        lower = (n + 0.5) * np.pi

        if a == 0.0:
            d = 0.0
        else:
            upper_d = np.nextafter(0.5 * np.pi, 0.0)

            def equation(d):
                return (lower + d) * np.tan(d) - a

            d = brentq(equation, 0.0, upper_d, xtol=2e-14, rtol=1e-14)

        q = lower + d
        energy = 2.0 * q * q / (mass * length * length)

        # For an even state, normalization on both half-intervals gives
        # A = 1 - sin(2q)/(2q), and the wall pressure is (2E/L)/A.
        normalization_factor = 1.0 - np.sin(2.0 * q) / (2.0 * q)
        wall_force = (2.0 * energy / length) / normalization_factor

        energies.append(energy)
        wall_forces.append(wall_force)

        # Odd states vanish at x=0 and are unaffected by the delta defect.
        q_odd = (n + 1.0) * np.pi
        odd_energy = 2.0 * q_odd * q_odd / (mass * length * length)
        energies.append(odd_energy)
        wall_forces.append(2.0 * odd_energy / length)

    order = np.argsort(energies)
    return (np.asarray(energies, dtype=float)[order],
            np.asarray(wall_forces, dtype=float)[order])


def levels(length, strength, mass, count=24):
    """Return the lowest eigenenergies.

    This helper is retained for compatibility with the original module.
    """
    return _spectrum(float(length), float(strength), float(mass), int(count))[0]


def force(experiment, mass):
    """Return the canonical mean force for one experiment."""
    length = float(experiment["length"])
    strength = float(experiment["strength"])
    temperature = float(experiment["temperature"])
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    energy, state_force = _spectrum(length, strength, float(mass))

    # Subtract the ground-state energy before exponentiating.  This is both
    # numerically stable at low temperature and leaves the Gibbs weights
    # unchanged.
    weights = np.exp(-(energy - energy[0]) / temperature)
    weights /= weights.sum()
    return float(weights @ state_force)


def predict_at(experiments, mass):
    """Predict in the same order as the supplied experiments."""
    return np.asarray([force(experiment, mass) for experiment in experiments],
                      dtype=float)


class Model:
    def __init__(self):
        self.mass = None

    def fit(self, records):
        """Fit the common mass by weighted least squares against calibration."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([float(record["value"]) for record in records])
        sigma = np.asarray([float(record.get("sigma", 1.0)) for record in records])
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("calibration values and sigmas must be finite; sigmas positive")

        weights = 1.0 / (sigma * sigma)

        def objective(mass):
            residual = predict_at(experiments, mass) - values
            return float(np.dot(weights, residual * residual))

        result = minimize_scalar(
            objective,
            bounds=(_MASS_MIN, _MASS_MAX),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 200},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError("mass fit did not converge")

        self.mass = float(result.x)
        return self

    def predict(self, experiments):
        if self.mass is None:
            raise RuntimeError("Model.fit must be called before predict")
        return predict_at(experiments, self.mass)
