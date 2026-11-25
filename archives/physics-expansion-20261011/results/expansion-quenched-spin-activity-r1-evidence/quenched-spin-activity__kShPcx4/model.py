"""Thermodynamic counting-field model for the interacting magnet."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar


_MIN_ATTEMPT_RATE = 0.8
_MAX_ATTEMPT_RATE = 1.4


def _sech(x):
    return 1.0 / np.cosh(x)


def _one_population_value(magnetization, tanh_field, tilted_sech):
    z = np.clip(1.0 - magnetization * magnetization, 0.0, None)
    return tanh_field * magnetization + tilted_sech * np.sqrt(z)


def _best_split(total_magnetization, tanh_fields, tilted_seches):
    """Maximize over the two population magnetizations at fixed total m."""
    m = float(total_magnetization)
    t_plus, t_minus = tanh_fields
    c_plus, c_minus = tilted_seches

    if m <= -1.0 + 1e-13 or m >= 1.0 - 1e-13:
        x = y = np.clip(m, -1.0, 1.0)
        return _one_population_value(x, t_plus, c_plus) + _one_population_value(
            y, t_minus, c_minus
        )

    lower = max(-1.0, 2.0 * m - 1.0)
    upper = min(1.0, 2.0 * m + 1.0)

    def derivative(x):
        y = 2.0 * m - x
        dx = max(1.0 - x * x, 1e-300)
        dy = max(1.0 - y * y, 1e-300)
        return (
            t_plus
            - c_plus * x / np.sqrt(dx)
            - t_minus
            + c_minus * y / np.sqrt(dy)
        )

    width = upper - lower
    eps = min(1e-10, width * 1e-6)
    x = brentq(derivative, lower + eps, upper - eps, xtol=2e-14, rtol=1e-14)
    y = 2.0 * m - x
    return _one_population_value(x, t_plus, c_plus) + _one_population_value(
        y, t_minus, c_minus
    )


def _variational_shape(experiment):
    """Return psi divided by the attempt rate for one experiment."""
    coupling = float(experiment["coupling"])
    field = float(experiment["field"])
    disorder = float(experiment["disorder"])
    bias_plus = float(experiment["bias_plus"])
    bias_minus = float(experiment["bias_minus"])

    # On the calibration slice the populations are independent.  Besides
    # being exact, this closed form avoids doing a numerical maximization for
    # every repeated calibration record.
    if coupling == 0.0:
        local_fields = np.array([field + disorder, field - disorder])
        biases = np.array([bias_plus, bias_minus])
        t = np.tanh(local_fields)
        return float(
            0.25
            * np.sum(-1.0 + np.sqrt(t * t + np.exp(-2.0 * biases) * _sech(local_fields) ** 2))
        )

    def value_at_m(m):
        local_fields = np.array(
            [coupling * m + field + disorder, coupling * m + field - disorder]
        )
        tanh_fields = np.tanh(local_fields)
        tilted_seches = np.exp(-np.array([bias_plus, bias_minus])) * _sech(local_fields)
        split_value = _best_split(m, tanh_fields, tilted_seches)
        # Half the spins belong to each population and every rate has a 1/2.
        return 0.25 * (split_value - 2.0)

    # Locate all broad maxima on the compact interval, then refine each one.
    grid = np.linspace(-1.0, 1.0, 401)
    values = np.array([value_at_m(m) for m in grid])
    best = float(np.max(values))
    candidates = [0, len(grid) - 1]
    candidates.extend(
        i
        for i in range(1, len(grid) - 1)
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]
    )
    for i in candidates:
        if i == 0:
            lo, hi = grid[0], grid[1]
        elif i == len(grid) - 1:
            lo, hi = grid[-2], grid[-1]
        else:
            lo, hi = grid[i - 1], grid[i + 1]
        result = minimize_scalar(
            lambda m: -value_at_m(m),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 2e-12},
        )
        best = max(best, -float(result.fun))
    return best


def response(e, kappa):
    """Evaluate the limiting SCGF for one experiment and attempt rate."""
    return float(kappa) * _variational_shape(e)


def predict_at(experiments, attempt_rate):
    return np.asarray(
        [response(experiment, float(attempt_rate)) for experiment in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.attempt_rate = None

    def fit(self, records):
        """Fit the common attempt rate by inverse-variance weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        shapes = np.asarray(
            [_variational_shape(record["input"]) for record in records], dtype=float
        )
        values = np.asarray([float(record["value"]) for record in records], dtype=float)
        sigmas = np.asarray([float(record["sigma"]) for record in records], dtype=float)
        if not np.all(np.isfinite(shapes)) or not np.all(np.isfinite(values)):
            raise ValueError("calibration records must contain finite values")
        if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            raise ValueError("calibration sigma values must be positive and finite")

        weights = 1.0 / (sigmas * sigmas)
        denominator = np.sum(weights * shapes * shapes)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("calibration records do not identify an attempt rate")
        estimate = np.sum(weights * shapes * values) / denominator
        self.attempt_rate = float(np.clip(estimate, _MIN_ATTEMPT_RATE, _MAX_ATTEMPT_RATE))
        return self

    def predict(self, experiments):
        if self.attempt_rate is None:
            raise RuntimeError("Model must be fitted before prediction")
        return predict_at(experiments, self.attempt_rate)
