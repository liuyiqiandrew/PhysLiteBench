import numpy as np


def _amplitude(density, time_a, time_b):
    """Known (D-independent) shape factor of the limiting covariance coefficient.

    The tagged particle in the infinite single-file system is a fractional
    Brownian motion of Hurst index 1/4, so that
        Cov[X(ta), X(tb)] = sqrt(D/(2*pi)) / density
                            * (sqrt(ta) + sqrt(tb) - sqrt(|ta - tb|)).
    The coefficient is linear in sqrt(D); this returns everything but sqrt(D).
    """
    ta = float(time_a)
    tb = float(time_b)
    density = float(density)
    return (np.sqrt(ta) + np.sqrt(tb) - np.sqrt(abs(ta - tb))) / (
        density * np.sqrt(2.0 * np.pi)
    )


def covariance(diffusivity, density, time_a, time_b):
    """Limiting coefficient Cov[X(lambda*ta), X(lambda*tb)] / sqrt(lambda)."""
    return np.sqrt(float(diffusivity)) * _amplitude(density, time_a, time_b)


def predict_at(experiments, diffusivity):
    """Vectorized evaluation of ``covariance`` over an iterable of experiments."""
    experiments = list(experiments)
    if not experiments:
        return np.zeros(0, dtype=float)

    density = np.array([float(e["density"]) for e in experiments], dtype=float)
    time_a = np.array([float(e["time_a"]) for e in experiments], dtype=float)
    time_b = np.array([float(e["time_b"]) for e in experiments], dtype=float)

    amp = (np.sqrt(time_a) + np.sqrt(time_b) - np.sqrt(np.abs(time_a - time_b))) / (
        density * np.sqrt(2.0 * np.pi)
    )
    return np.sqrt(float(diffusivity)) * amp


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Infer the common diffusivity from calibration records.

        With a known constant instrument sigma, the model is linear in
        ``sqrt(D)``; the maximum-likelihood estimate is the inverse-variance
        weighted least-squares solution, and ``D`` is its square.
        """
        records = list(records)
        if not records:
            raise ValueError("fit requires at least one calibration record.")

        amp = np.array(
            [
                _amplitude(
                    r["input"]["density"], r["input"]["time_a"], r["input"]["time_b"]
                )
                for r in records
            ],
            dtype=float,
        )
        y = np.array([float(r["value"]) for r in records], dtype=float)
        sigma = np.array(
            [float(r.get("sigma", 1.0)) for r in records], dtype=float
        )

        weights = np.where(np.isfinite(sigma) & (sigma > 0.0), 1.0 / sigma**2, 0.0)
        denom = float(np.sum(weights * amp * amp))
        if denom <= 0.0:
            raise ValueError("Calibration records give no leverage on diffusivity.")

        sqrt_d = float(np.sum(weights * amp * y) / denom)
        self.diffusivity = max(sqrt_d, 0.0) ** 2
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
