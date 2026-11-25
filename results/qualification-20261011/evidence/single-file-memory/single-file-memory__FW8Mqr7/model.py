import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Return the limiting covariance coefficient for one experiment.

    The long-time covariance of a tagged particle in a single file is
    proportional to ``sqrt(diffusivity)``.  Keeping the calculation in a
    separate function makes the scale parameter used by ``fit`` explicit.
    """
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)

    return (
        np.sqrt(diffusivity / (2.0 * np.pi)) / density
        * (np.sqrt(time_a) + np.sqrt(time_b) - np.sqrt(abs(time_a - time_b)))
    )


def predict_at(experiments, diffusivity):
    if diffusivity is None or not np.isfinite(diffusivity) or diffusivity <= 0:
        raise ValueError("Model must be fitted before prediction.")

    predictions = [
        covariance(
            diffusivity,
            e["density"],
            e["time_a"],
            e["time_b"],
        )
        for e in experiments
    ]
    result = np.asarray(predictions, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("Experiments produced non-finite predictions.")
    return result


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Infer the common diffusivity from noisy calibration records.

        For record ``i``, the apparatus model has the form

            value_i = sqrt(diffusivity) * coefficient_i + error_i.

        Thus the weighted least-squares estimate can be obtained directly,
        without a numerical optimizer.  The documented physical bounds are
        applied to the resulting estimate.
        """
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")

        coefficients = []
        values = []
        weights = []
        for record in records:
            experiment = record["input"]
            value = float(record["value"])
            sigma = float(record["sigma"])
            if not np.isfinite(value) or not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("Calibration values and sigmas must be finite; sigma must be positive.")

            coefficient = covariance(
                1.0,
                experiment["density"],
                experiment["time_a"],
                experiment["time_b"],
            )
            if not np.isfinite(coefficient) or coefficient <= 0:
                raise ValueError("Calibration inputs must produce a positive finite coefficient.")

            coefficients.append(coefficient)
            values.append(value)
            weights.append(1.0 / sigma**2)

        coefficients = np.asarray(coefficients, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)

        # Fit alpha = sqrt(D), then square it.  This is the linear parameter
        # in the observation model and correctly incorporates each sigma.
        denominator = np.sum(weights * coefficients**2)
        alpha = np.sum(weights * coefficients * values) / denominator
        if not np.isfinite(alpha):
            raise ValueError("Calibration records do not determine a finite diffusivity.")

        # Constrain the fitted linear parameter before squaring so the
        # estimate is also correct for an unusually noisy calibration set.
        alpha = float(np.clip(alpha, np.sqrt(0.8), np.sqrt(1.2)))
        self.diffusivity = alpha**2
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
