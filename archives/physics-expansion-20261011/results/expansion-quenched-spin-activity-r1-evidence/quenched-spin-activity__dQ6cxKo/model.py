"""Thermodynamic counting-statistics model for the switching apparatus.

The tilted generator is used here in its symmetric representation.  If
``m_plus`` and ``m_minus`` are the magnetizations of the two populations,
its largest eigenvalue per spin is the maximum of a two-dimensional symbol.
The reduction to this symbol is exact in the N -> infinity limit specified
by the problem (the long-time limit is taken first).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def _sech(x: float) -> float:
    """Return sech(x), with the relevant parameter range kept stable."""

    return 1.0 / np.cosh(x)


def _independent_coefficient(experiment: Mapping[str, float]) -> float:
    """Coefficient of the attempt rate when coupling is zero."""

    h = float(experiment["field"])
    disorder = float(experiment["disorder"])
    biases = (float(experiment["bias_plus"]), float(experiment["bias_minus"]))
    fields = (h + disorder, h - disorder)

    value = 0.0
    for local_field, bias in zip(fields, biases):
        t = np.tanh(local_field)
        c = _sech(local_field)
        value += -1.0 + np.sqrt(t * t + np.exp(-2.0 * bias) * c * c)
    return 0.25 * value


def _inner_maximum(
    magnetization: float,
    coupling: float,
    field: float,
    disorder: float,
    bias_plus: float,
    bias_minus: float,
) -> float:
    """Maximize the symbol at fixed total magnetization.

    At fixed ``m=(m_plus+m_minus)/2`` the objective is strictly concave in
    ``m_plus``.  Solving its monotone first-order condition is both quicker
    and more reliable than a generic two-dimensional optimizer.
    """

    m = float(magnetization)
    plus_field = coupling * m + field + disorder
    minus_field = coupling * m + field - disorder
    t_plus, t_minus = np.tanh(plus_field), np.tanh(minus_field)
    a_plus = np.exp(-bias_plus) * _sech(plus_field)
    a_minus = np.exp(-bias_minus) * _sech(minus_field)

    lower = max(-1.0, 2.0 * m - 1.0)
    upper = min(1.0, 2.0 * m + 1.0)

    if upper - lower <= 1.0e-14:
        x = lower
    else:
        def derivative(x: float) -> float:
            y = 2.0 * m - x

            if abs(x) >= 1.0:
                first = np.inf if x < 0.0 else -np.inf
            else:
                first = -a_plus * x / np.sqrt(1.0 - x * x)

            if abs(y) >= 1.0:
                second = np.inf if y > 0.0 else -np.inf
            else:
                second = a_minus * y / np.sqrt(1.0 - y * y)
            return t_plus - t_minus + first + second

        x = brentq(derivative, lower, upper, xtol=2.0e-14, rtol=1.0e-14)

    y = 2.0 * m - x
    root_x = np.sqrt(max(0.0, 1.0 - x * x))
    root_y = np.sqrt(max(0.0, 1.0 - y * y))
    return x * t_plus + a_plus * root_x + y * t_minus + a_minus * root_y


def _symbol_maximum(experiment: Mapping[str, float]) -> float:
    """Return the attempt-rate-normalized thermodynamic SCGF."""

    coupling = float(experiment["coupling"])
    field = float(experiment["field"])
    disorder = float(experiment["disorder"])
    bias_plus = float(experiment["bias_plus"])
    bias_minus = float(experiment["bias_minus"])

    if bias_plus == 0.0 and bias_minus == 0.0:
        return 0.0

    def objective(m: float) -> float:
        return _inner_maximum(
            m, coupling, field, disorder, bias_plus, bias_minus
        )

    # The outer problem can have two symmetry-related maxima in the
    # ferromagnetic regime.  A fixed grid locates every such basin; bounded
    # one-dimensional refinements give the required precision.
    grid = np.linspace(-1.0, 1.0, 161)
    values = np.asarray([objective(m) for m in grid])
    candidates = [float(values[0]), float(values[-1])]

    for index in range(1, len(grid) - 1):
        if values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            result = minimize_scalar(
                lambda m: -objective(float(m)),
                bounds=(float(grid[index - 1]), float(grid[index + 1])),
                method="bounded",
                options={"xatol": 2.0e-11, "maxiter": 100},
            )
            candidates.append(float(-result.fun))

    candidates.append(float(np.max(values)))
    return 0.25 * (max(candidates) - 2.0)


def response(e: Mapping[str, float], kappa: float) -> float:
    """Predict ``psi`` for one experiment at the supplied attempt rate."""

    return float(kappa) * _symbol_maximum(e)


def predict_at(experiments: Iterable[Mapping[str, float]], attempt_rate: float) -> np.ndarray:
    """Vectorized public prediction helper, preserving input order."""

    return np.asarray(
        [response(experiment, float(attempt_rate)) for experiment in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.attempt_rate = None

    def fit(self, records):
        """Fit the common attempt rate by weighted least squares."""

        coefficients = []
        values = []
        weights = []
        for record in records:
            experiment = record["input"]
            coefficients.append(_independent_coefficient(experiment))
            values.append(float(record["value"]))
            sigma = float(record["sigma"])
            weights.append(1.0 / (sigma * sigma))

        coefficients = np.asarray(coefficients, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)

        denominator = np.sum(weights * coefficients * coefficients)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("Calibration records do not identify attempt_rate")

        estimate = np.sum(weights * coefficients * values) / denominator
        self.attempt_rate = float(np.clip(estimate, 0.8, 1.4))
        return self

    def predict(self, experiments):
        if self.attempt_rate is None:
            raise RuntimeError("Model must be fitted before prediction")
        return predict_at(experiments, self.attempt_rate)
