from functools import lru_cache
import numpy as np


def _rates(experiment, friction):
    return _rates_at(experiment['temperature'], experiment['contrast'], experiment['wavenumber'], friction)


@lru_cache(256)
def _rates_at(base, contrast, wave, friction):
    x = np.arange(512)*2*np.pi/512
    temperature = base*(1+contrast*np.cos(x))
    gradient = -base*contrast*wave*np.sin(x)
    density = 1/temperature
    density /= np.sum(density)
    local_rate = gradient**2/(2*friction*temperature)
    mean = float(density@local_rate)
    rhs = -friction*(local_rate-mean)/(wave**2*temperature)
    modes = np.fft.fftfreq(len(x), 1/len(x))
    transformed = np.fft.fft(rhs)
    potential = np.zeros(len(x), dtype=complex)
    potential[1:] = -transformed[1:]/modes[1:]**2
    potential = np.fft.ifft(potential).real
    potential -= density@potential
    variance = float(2*density@((local_rate-mean)*potential))
    return mean, variance


class Model:
    def __init__(self):
        self.friction = 1.0

    def fit(self, records):
        """Fit the (constant) friction coefficient from calibration records.

        Both rates returned by ``_rates_at`` are proportional to the inverse
        friction.  If ``c_i`` denotes the rate predicted at unit friction,
        the calibration model is therefore

            value_i = c_i / friction + noise_i.

        Solving the weighted least-squares problem in ``q = 1/friction`` is
        linear and avoids a numerical optimizer or sensitivity to the initial
        guess.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        coefficients = []
        values = []
        weights = []
        for record in records:
            experiment = record["input"]
            statistic = experiment["statistic"]
            if statistic not in ("mean", "variance"):
                raise ValueError(f"unknown statistic: {statistic!r}")

            mean, variance = _rates(experiment, 1.0)
            coefficients.append(mean if statistic == "mean" else variance)
            values.append(float(record["value"]))
            sigma = float(record["sigma"])
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("record sigma must be finite and positive")
            weights.append(1.0 / sigma**2)

        coefficients = np.asarray(coefficients, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if not np.isfinite(coefficients).all() or not np.isfinite(values).all():
            raise ValueError("calibration records must contain finite values")

        # Weighted least squares for value ~= coefficient * q, q=1/friction.
        denominator = np.sum(weights * coefficients**2)
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("calibration records do not identify friction")
        inverse_friction = np.sum(weights * coefficients * values) / denominator
        if not np.isfinite(inverse_friction) or inverse_friction <= 0:
            raise ValueError("calibration records imply invalid friction")

        # The apparatus specification bounds the unknown friction.  Clipping
        # keeps fits based on unusual/noisy subsets inside that physical range.
        self.friction = float(np.clip(1.0 / inverse_friction, 0.7, 1.6))
        return self

    def predict(self, experiments):
        values = []
        for e in experiments:
            mean, variance = _rates(e, self.friction)
            if e['statistic'] == 'mean':
                values.append(mean)
            elif e['statistic'] == 'variance':
                values.append(variance)
            else:
                raise ValueError(f"unknown statistic: {e['statistic']!r}")
        return np.asarray(values, dtype=float)
