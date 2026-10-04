"""Prediction model for the small-mass entropy rates."""

from functools import lru_cache

import numpy as np


_GRID_SIZE = 2048


def _rates(experiment, friction):
    return _rates_at(
        float(experiment["temperature"]),
        float(experiment["contrast"]),
        int(experiment["wavenumber"]),
        float(friction),
    )


@lru_cache(maxsize=512)
def _rates_at(base, contrast, wave, friction):
    """Return the mean and centered variance rates for one experiment."""

    n = _GRID_SIZE
    x = 2.0 * np.pi * np.arange(n) / n
    angle = wave * x
    temperature = base * (1.0 + contrast * np.cos(angle))
    gradient = -base * contrast * wave * np.sin(angle)
    curvature = -base * contrast * wave * wave * np.cos(angle)

    # The stationary position density is proportional to 1/T.
    weight = 1.0 / temperature
    weight /= np.mean(weight)
    mean_coefficient = float(
        np.mean(weight * gradient * gradient / (2.0 * temperature))
    )
    mean = mean_coefficient / friction
    if mean_coefficient == 0.0:
        return mean, 0.0

    # First-order term of the tilted homogenized generator.
    u_term = -1.5 * curvature + 2.0 * gradient * gradient / temperature
    u_mean = float(np.mean(weight * u_term))

    # Solve the periodic Poisson equation for the first-order eigenfunction
    # correction.  Frequencies are with respect to x in [0, 2*pi).
    rhs = (u_mean - u_term) / temperature
    frequencies = np.fft.fftfreq(n, d=1.0 / n)
    rhs_hat = np.fft.fft(rhs)
    correction_hat = np.zeros(n, dtype=complex)
    correction_hat[1:] = -rhs_hat[1:] / (frequencies[1:] ** 2)
    correction = np.fft.ifft(correction_hat).real
    correction -= np.mean(weight * correction)
    correction_gradient = np.fft.ifft(
        1j * frequencies * np.fft.fft(correction)
    ).real

    # Fast velocity Green-Kubo term plus the slow positional correction.
    fast_variance = 11.0 * gradient * gradient / (4.0 * temperature)
    positional_correction = (
        -3.0 * gradient * correction_gradient + u_term * correction
    )
    variance_coefficient = 2.0 * float(
        np.mean(weight * (fast_variance + positional_correction))
    )
    variance = max(0.0, variance_coefficient / friction)
    return mean, variance


class Model:
    def __init__(self):
        self.friction = 1.0

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        coefficients = []
        values = []
        weights = []
        for record in records:
            experiment = record["input"]
            mean, variance = _rates(experiment, 1.0)
            statistic = experiment["statistic"]
            if statistic == "mean":
                coefficient = mean
            elif statistic == "variance":
                coefficient = variance
            else:
                raise ValueError(f"unknown statistic: {statistic!r}")
            sigma = float(record.get("sigma", 1.0))
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("record sigma must be positive and finite")
            coefficients.append(coefficient)
            values.append(float(record["value"]))
            weights.append(1.0 / (sigma * sigma))

        coefficients = np.asarray(coefficients, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = float(np.sum(weights * coefficients * coefficients))
        numerator = float(np.sum(weights * coefficients * values))
        if denominator <= 0.0 or not np.isfinite(numerator):
            raise ValueError("calibration records do not identify friction")
        inverse_friction = numerator / denominator
        if not np.isfinite(inverse_friction) or inverse_friction <= 0.0:
            raise ValueError("calibration records imply nonpositive friction")

        self.friction = float(np.clip(1.0 / inverse_friction, 0.7, 1.6))
        return self

    def predict(self, experiments):
        values = []
        for e in experiments:
            mean, variance = _rates(e, self.friction)
            if e["statistic"] == "mean":
                values.append(mean)
            elif e["statistic"] == "variance":
                values.append(variance)
            else:
                raise ValueError(f"unknown statistic: {e['statistic']!r}")
        return np.asarray(values, dtype=float).reshape(-1)
