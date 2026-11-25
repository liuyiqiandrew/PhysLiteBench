"""Linear lowest-thickness-mode model for the magnetic bilayer.

The state ordering used here is ``(m_y0, m_z0, m_y1, m_z1)``.  All of
these are complex Fourier amplitudes for the supplied (positive or
negative) in-plane wavevector.
"""

from functools import lru_cache

import numpy as np


DAMPING = 0.04
GYRO_BOUNDS = (0.7, 1.3)
ROTATION = np.kron(np.eye(2), np.array([[0.0, -1.0], [1.0, 0.0]]))


def energy_matrix(kx, ky, gap):
    """Return the quadratic transverse-energy matrix for one wavevector.

    The diagonal terms are the applied field plus anisotropy, exchange, and
    the self-demagnetizing terms.  The imaginary off-diagonal terms are the
    magnetostatic coupling between in-plane and out-of-plane components in
    different films.  They are required when ``ky`` is nonzero: they encode
    the phase accumulated by the Fourier representation of the stray field.
    """
    k = float(np.hypot(kx, ky))
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("wavevector must have a finite, nonzero magnitude")

    direction = float(ky) / k
    one_minus_exp = -np.expm1(-k)
    p = one_minus_exp / k
    q = np.exp(-k * float(gap)) * one_minus_exp**2 / (2.0 * k)

    # Demagnetizing matrix, with state order (y0,z0,y1,z1).  For the
    # convention Re[a exp(i k.r)], the y-z inter-film entries are imaginary.
    demag = np.zeros((4, 4), dtype=complex)
    demag[np.ix_([0, 2], [0, 2])] = direction**2 * np.array(
        [[1.0 - p, q], [q, 1.0 - p]]
    )
    demag[np.ix_([1, 3], [1, 3])] = np.array([[p, -q], [-q, p]])

    cross = 1j * direction * q
    demag[0, 3] = -cross
    demag[3, 0] = cross
    demag[1, 2] = -cross
    demag[2, 1] = cross

    # H_ext + K is 0.4 + 0.05 in film 0 and 0.4 + 0.30 in film 1.
    stiffness = np.repeat([0.45, 0.70], 2) + 0.08 * k**2
    return demag + np.diag(stiffness)


@lru_cache(maxsize=512)
def mode_basis(kx, ky, gap):
    """Cache the gyro-rate-independent eigenbasis of the linear dynamics."""
    energy = energy_matrix(kx, ky, gap)
    generator = (ROTATION - DAMPING * np.eye(4)) @ energy / (1.0 + DAMPING**2)
    rates, vectors = np.linalg.eig(generator)
    inverse = np.linalg.inv(vectors)
    return rates, vectors, inverse


def predict_at(experiments, gyro_rate):
    """Evaluate experiments in their input order for one gyro rate."""
    gyro_rate = float(gyro_rate)
    if not np.isfinite(gyro_rate):
        raise ValueError("gyro_rate must be finite")

    result = []
    for experiment in experiments:
        kx, ky = experiment["wavevector"]
        rates, vectors, inverse = mode_basis(float(kx), float(ky), float(experiment["gap"]))

        initial = np.zeros(4, dtype=complex)
        initial[[0, 2]] = (
            np.asarray(experiment["initial_real"], dtype=float)
            + 1j * np.asarray(experiment["initial_imag"], dtype=float)
        )
        coefficients = inverse @ initial
        state = vectors @ (np.exp(rates * gyro_rate * float(experiment["time"])) * coefficients)

        value = state[2 * int(experiment["layer"]) + 1]
        result.append(value.real if experiment["quadrature"] == "real" else value.imag)

    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.gyro_rate = None

    def fit(self, records):
        """Fit the common gyromagnetic rate by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        measured = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if (
            not np.isfinite(measured).all()
            or not np.isfinite(sigma).all()
            or np.any(sigma <= 0.0)
        ):
            raise ValueError("calibration values and sigmas must be finite; sigmas must be positive")

        def objective(rate):
            residual = (predict_at(experiments, rate) - measured) / sigma
            return float(residual @ residual)

        # A coarse scan makes the one-dimensional fit robust to oscillatory
        # observations.  Golden-section refinement then gives more precision
        # than is warranted by the calibration noise.
        grid = np.linspace(GYRO_BOUNDS[0], GYRO_BOUNDS[1], 1201)
        values = np.asarray([objective(rate) for rate in grid])
        best = int(np.argmin(values))
        if best == 0 or best == len(grid) - 1:
            fitted = grid[best]
        else:
            left, right = grid[best - 1], grid[best + 1]
            golden = (np.sqrt(5.0) - 1.0) / 2.0
            x1 = right - golden * (right - left)
            x2 = left + golden * (right - left)
            f1, f2 = objective(x1), objective(x2)
            for _ in range(60):
                if f1 <= f2:
                    right, x2, f2 = x2, x1, f1
                    x1 = right - golden * (right - left)
                    f1 = objective(x1)
                else:
                    left, x1, f1 = x1, x2, f2
                    x2 = left + golden * (right - left)
                    f2 = objective(x2)
            fitted = 0.5 * (left + right)

        self.gyro_rate = float(fitted)
        return self

    def predict(self, experiments):
        if self.gyro_rate is None:
            raise RuntimeError("Model.fit must be called before predict")
        return predict_at(experiments, self.gyro_rate)
