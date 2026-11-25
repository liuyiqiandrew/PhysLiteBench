"""Effective model for the narrow-channel spectrometer.

In the zero-width limit only the lowest transverse mode remains at finite
energy. Projecting the Cartesian kinetic energy onto that mode gives a
one-dimensional Dirichlet Hamiltonian on ``(0, 1)``. The transverse ground
energy is cancelled by the applied potential in the problem statement; the
variation of the transverse mode leaves the geometric potential implemented
below.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_legendre


_GEOMETRIC_COEFFICIENT = (np.pi**2 + 3.0) / 12.0


def gap(experiment, modes=80, points=320):
    """Return the un-gained gap for one experiment."""
    amplitude = float(experiment["amplitude"])
    harmonic = float(experiment["harmonic"])
    phase = float(experiment["phase"])
    well = float(experiment["well"])
    level = int(experiment["level"])

    if level < 0 or modes < level + 1:
        raise ValueError("invalid level or insufficient basis modes")

    quadrature_nodes, quadrature_weights = roots_legendre(points)
    x = (quadrature_nodes + 1.0) / 2.0
    weights = quadrature_weights / 2.0

    n = np.arange(1, modes + 1, dtype=float)
    basis = np.sqrt(2.0) * np.sin(np.pi * x[:, None] * n)

    # log(a(x)) = amplitude*cos(2*pi*x) +
    #              harmonic*cos(4*pi*x + phase).
    log_width_derivative = (
        -2.0 * np.pi * amplitude * np.sin(2.0 * np.pi * x)
        - 4.0 * np.pi * harmonic * np.sin(4.0 * np.pi * x + phase)
    )
    potential = (
        well * np.cos(2.0 * np.pi * x)
        + _GEOMETRIC_COEFFICIENT * log_width_derivative**2
    )

    matrix = np.diag((np.pi * n) ** 2)
    matrix += basis.T @ ((weights * potential)[:, None] * basis)
    values = eigh(matrix, eigvals_only=True, subset_by_index=(0, level))
    return float(values[level] - values[0])


def predict_at(experiments, gain):
    """Evaluate gained gaps in the input order."""
    if gain is None:
        raise RuntimeError("fit the model before calling predict")
    return float(gain) * np.asarray(
        [gap(experiment) for experiment in experiments], dtype=float
    )


class Model:
    def __init__(self):
        self.gain = None

    def fit(self, records):
        """Estimate the common spectrometer gain by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        gaps = np.asarray([gap(record["input"]) for record in records], dtype=float)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if not np.all(np.isfinite(gaps)) or not np.all(np.isfinite(values)):
            raise ValueError("calibration data must be finite")
        if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            raise ValueError("calibration uncertainties must be positive and finite")

        weights = 1.0 / sigmas**2
        denominator = np.sum(weights * gaps**2)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("calibration records do not identify a gain")
        estimate = np.sum(weights * gaps * values) / denominator
        # The stated apparatus constrains the gain to this interval.
        self.gain = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.gain)
