"""Effective model for the narrow, variable-width channel."""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_legendre


_GEOMETRIC_COEFFICIENT = 0.25 + np.pi**2 / 12.0


@lru_cache(maxsize=16)
def _basis_data(modes, points):
    q, weights = roots_legendre(points)
    x = (q + 1.0) / 2.0
    weights = weights / 2.0
    n = np.arange(1, modes + 1, dtype=float)
    basis = np.sqrt(2.0) * np.sin(np.pi * x[:, None] * n)
    return x, weights, n, basis


def gap(experiment, modes=80, points=320):
    x, weights, n, basis = _basis_data(int(modes), int(points))

    amplitude = float(experiment['amplitude'])
    harmonic = float(experiment['harmonic'])
    phase = float(experiment['phase'])
    well = float(experiment['well'])
    level = int(experiment['level'])

    # d(log(a))/dx for the width specified in the README.  The coefficient
    # below is the norm squared of the x derivative of the centered
    # transverse ground state.
    log_width_derivative = (
        -2.0*np.pi*amplitude*np.sin(2.0*np.pi*x)
        -4.0*np.pi*harmonic*np.sin(4.0*np.pi*x + phase)
    )
    potential = (
        well*np.cos(2.0*np.pi*x)
        + _GEOMETRIC_COEFFICIENT*log_width_derivative**2
    )
    matrix = np.diag((np.pi*n)**2)
    matrix = matrix + basis.T@((weights*potential)[:, None]*basis)
    values = eigh(matrix, eigvals_only=True,
                  subset_by_index=(0, level), check_finite=False)
    return float(values[-1]-values[0])


def predict_at(experiments, gain):
    return float(gain)*np.asarray([gap(e) for e in experiments])


class Model:
    def __init__(self):
        self.gain = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')
        theoretical = np.asarray([gap(r['input']) for r in records])
        measured = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not (np.all(np.isfinite(theoretical)) and
                np.all(np.isfinite(measured))):
            raise ValueError('calibration values must be finite')
        if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
            raise ValueError('calibration uncertainties must be positive')
        weights = 1.0/sigma**2
        gain = np.sum(weights*theoretical*measured)/np.sum(weights*theoretical**2)
        self.gain = float(np.clip(gain, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.gain is None:
            raise RuntimeError('Model must be fit before prediction')
        return predict_at(experiments, self.gain)
