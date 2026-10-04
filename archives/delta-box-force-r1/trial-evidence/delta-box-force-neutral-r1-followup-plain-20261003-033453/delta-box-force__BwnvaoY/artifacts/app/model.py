"""Model for the force on a wall of a one-dimensional delta-defect cavity."""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from functools import lru_cache


_DEFAULT_LEVELS = 32


@lru_cache(maxsize=4096)
def _spectrum(length, strength, mass, count=_DEFAULT_LEVELS):
    """Return energies and wall forces for the lowest cavity states.

    Put q = k*length/2.  The odd states do not see the defect and have
    q=(n+1)pi.  The even states obey

        q*tan(q-(n+1/2)pi) = mass*strength*length/2.

    The force of an eigenstate is the stress at the right hard wall,
    |psi'(length/2)|^2/(2*mass).  The defect changes the normalization of
    the even states, so this is not generally 2*energy/length.
    """
    length = float(length)
    strength = float(strength)
    mass = float(mass)
    count = int(count)
    if length <= 0.0 or mass <= 0.0 or strength < 0.0 or count <= 0:
        raise ValueError("invalid cavity parameters")

    coefficient = 1.0 / (2.0 * mass)
    ratio = mass * strength * length / 2.0
    energies = []
    forces = []

    for n in range(count):
        lower = (n + 0.5) * np.pi
        upper = (n + 1.0) * np.pi

        if ratio == 0.0:
            q = lower
        else:
            offset = brentq(
                lambda delta: (lower + delta) * np.tan(delta) - ratio,
                0.0,
                np.pi / 2.0 - 1.0e-13,
                xtol=2.0e-14,
                rtol=4.0 * np.finfo(float).eps,
            )
            q = lower + offset

        energy = coefficient * (2.0 * q / length) ** 2
        # Normalization correction for an even state; it is one when there
        # is no defect.  The same expression follows from -dE/dlength.
        offset = q - lower
        normalization = 1.0 + np.sin(2.0 * offset) / (2.0 * q)
        wall_force = 4.0 * q * q / (mass * length**3 * normalization)
        energies.append(energy)
        forces.append(wall_force)

        # Odd-parity state, unaffected by the delta potential.
        q = upper
        energies.append(coefficient * (2.0 * q / length) ** 2)
        forces.append(4.0 * q * q / (mass * length**3))

    order = np.argsort(energies)
    return np.asarray(energies, dtype=float)[order], np.asarray(forces, dtype=float)[order]


def levels(length, strength, mass, count=16):
    """Backward-compatible energy-only spectrum helper."""
    return _spectrum(float(length), float(strength), float(mass), int(count))[0]


def force(experiment, mass):
    """Return the canonical mean force on the right wall."""
    length = float(experiment["length"])
    strength = float(experiment["strength"])
    temperature = float(experiment["temperature"])
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    energy, wall_force = _spectrum(length, strength, float(mass), _DEFAULT_LEVELS)
    weight = np.exp(-(energy - energy[0]) / temperature)
    weight /= np.sum(weight)
    return float(weight @ wall_force)


def predict_at(experiments, mass):
    return np.asarray([force(experiment, mass) for experiment in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mass=None

    def fit(self, records):
        """Fit the common mass by weighted least squares on calibration data."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record.get("sigma", 1.0) for record in records], dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
            raise ValueError("calibration values and uncertainties must be finite and valid")

        def objective(mass):
            residual = (predict_at(experiments, mass) - values) / sigma
            return float(residual @ residual)

        result = minimize_scalar(
            objective,
            bounds=(0.4, 0.9),
            method="bounded",
            options={"xatol": 1.0e-12, "maxiter": 200},
        )
        self.mass = float(np.clip(result.x, 0.4, 0.9))
        return self

    def predict(self, experiments):
        if self.mass is None:
            raise RuntimeError("Model.fit must be called before Model.predict")
        return predict_at(experiments,self.mass)
