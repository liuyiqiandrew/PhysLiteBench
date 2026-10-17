import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time tagged-particle covariance for fixed initial positions.

    The initially deterministic density has no fluctuation contribution.
    Integrating the diffusive current noise gives
    sqrt(D/pi) / density * (sqrt(time_a + time_b) - sqrt(|time_a-time_b|)).
    Rationalizing the difference avoids cancellation for disparate times.
    """
    time_factor = 2.0 * np.minimum(time_a, time_b) / (
        np.sqrt(time_a + time_b) + np.sqrt(np.abs(time_a - time_b))
    )
    return np.sqrt(diffusivity / np.pi) / density * time_factor


def predict_at(experiments, diffusivity):
    return np.asarray([
        covariance(float(diffusivity), float(e["density"]),
                   float(e["time_a"]), float(e["time_b"]))
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Fit the common diffusivity by constrained Gaussian likelihood."""
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")

        basis = predict_at([r["input"] for r in records], 1.0)
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        if not (
            np.isfinite(basis).all()
            and (basis > 0).all()
            and np.isfinite(values).all()
            and np.isfinite(sigma).all()
            and (sigma > 0).all()
        ):
            raise ValueError("Calibration requires finite values and positive controls and sigma.")

        # Predictions are linear in sqrt(D), so weighted least squares has
        # an exact one-parameter solution. Enforce the documented D range.
        weighted_basis = basis / sigma
        weighted_values = values / sigma
        amplitude = np.dot(weighted_basis, weighted_values) / np.dot(
            weighted_basis, weighted_basis
        )
        amplitude = max(0.0, float(amplitude))
        self.diffusivity = float(np.clip(amplitude * amplitude, 0.8, 1.2))
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("Fit the model before requesting predictions.")
        return predict_at(experiments, self.diffusivity)
