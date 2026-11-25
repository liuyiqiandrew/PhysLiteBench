"""Model for the narrow-channel spectroscopy experiment.

In the narrow-channel limit only the transverse ground mode remains at finite
energy.  Projecting the Cartesian kinetic energy onto that mode gives the
one-dimensional Dirichlet operator

    -d2/dx2 + well*cos(2*pi*x)
        + (1/4 + pi**2/12) * (a'(x)/a(x))**2.

The last term is the geometric potential caused by the changing width.  The
explicit negative potential in the apparatus has already been used in taking
the limit: it cancels the transverse ground energy and must not be applied a
second time here.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_legendre


# Share quadrature and basis arrays across predictions.
_DEFAULT_MODES = 80
_DEFAULT_POINTS = 320
_q, _w = roots_legendre(_DEFAULT_POINTS)
_x = (_q + 1.0) / 2.0
_weights = _w / 2.0
_n = np.arange(1, _DEFAULT_MODES + 1, dtype=float)
_basis = np.sqrt(2.0) * np.sin(np.pi * _x[:, None] * _n[None, :])
_kinetic = np.diag((np.pi * _n) ** 2)
_GEOMETRIC_COEFFICIENT = 0.25 + np.pi**2 / 12.0


def _effective_potential(experiment, x):
    """Return the finite longitudinal potential in the width limit."""
    amplitude = float(experiment["amplitude"])
    harmonic = float(experiment["harmonic"])
    phase = float(experiment["phase"])
    well = float(experiment["well"])

    two_pi_x = 2.0 * np.pi * x
    log_width_derivative = (
        -2.0 * np.pi * amplitude * np.sin(two_pi_x)
        -4.0 * np.pi * harmonic * np.sin(2.0 * two_pi_x + phase)
    )
    return (
        well * np.cos(two_pi_x)
        + _GEOMETRIC_COEFFICIENT * log_width_derivative**2
    )


def gap(experiment, modes=_DEFAULT_MODES, points=_DEFAULT_POINTS):
    """Return the un-gained gap from the ground state to ``experiment.level``."""
    modes = int(modes)
    points = int(points)
    level = int(experiment["level"])
    if modes < 1 or points < 1:
        raise ValueError("modes and points must be positive")
    if level < 1 or level >= modes:
        raise ValueError("level must be between 1 and modes - 1")

    # Rebuild the basis only for callers requesting non-default resolution.
    if modes == _DEFAULT_MODES and points == _DEFAULT_POINTS:
        x, weights, basis, kinetic = _x, _weights, _basis, _kinetic
    else:
        q, legendre_weights = roots_legendre(points)
        x = (q + 1.0) / 2.0
        weights = legendre_weights / 2.0
        n = np.arange(1, modes + 1, dtype=float)
        basis = np.sqrt(2.0) * np.sin(np.pi * x[:, None] * n[None, :])
        kinetic = np.diag((np.pi * n) ** 2)

    potential = _effective_potential(experiment, x)
    matrix = kinetic + basis.T @ ((weights * potential)[:, None] * basis)
    values = eigh(matrix, eigvals_only=True, subset_by_index=(0, level))
    return float(values[level] - values[0])


def predict_at(experiments, gain):
    """Evaluate a sequence of experiments in input order."""
    if gain is None or not np.isfinite(gain):
        raise ValueError("the model must be fit before prediction")
    return float(gain) * np.asarray(
        [gap(experiment) for experiment in experiments], dtype=float
    )


class Model:
    def __init__(self):
        self.gain = None

    def fit(self, records):
        """Estimate the common gain using inverse-variance weighted LS."""
        design = []
        observed = []
        weights = []
        for record in records:
            sigma = float(record["sigma"])
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("record sigma must be positive and finite")
            design.append(gap(record["input"]))
            observed.append(float(record["value"]))
            weights.append(1.0 / sigma**2)

        if not design:
            raise ValueError("at least one calibration record is required")

        design = np.asarray(design, dtype=float)
        observed = np.asarray(observed, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = np.sum(weights * design**2)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("calibration records do not identify a gain")
        estimate = np.sum(weights * design * observed) / denominator
        self.gain = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        result = predict_at(experiments, self.gain)
        return np.asarray(result, dtype=float).reshape(-1)
