import math

import numpy as np


_MIN_DIFFUSIVITY = 0.8
_MAX_DIFFUSIVITY = 1.2


def covariance(diffusivity, density, time_a, time_b):
    return np.sqrt(diffusivity/(2*np.pi))/density * (np.sqrt(time_a)+np.sqrt(time_b)-np.sqrt(abs(time_a-time_b)))


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
        """Infer the common diffusivity by weighted least squares.

        The covariance is proportional to ``sqrt(diffusivity)``.  This makes
        the Gaussian maximum-likelihood fit a one-parameter linear regression
        in that square root, with the known record uncertainties as weights.
        """
        # Materialize once so that iterators are supported and empty input can
        # be rejected before changing the model's fitted state.
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")

        numerator_terms = []
        denominator_terms = []
        for record in records:
            try:
                experiment = record["input"]
                observed = float(record["value"])
                sigma = float(record["sigma"])
                density = float(experiment["density"])
                time_a = float(experiment["time_a"])
                time_b = float(experiment["time_b"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Each calibration record must contain a valid input, value, and sigma.") from exc

            values = (observed, sigma, density, time_a, time_b)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Calibration values and controls must be finite.")
            if sigma <= 0.0:
                raise ValueError("Calibration sigma must be positive.")
            if density <= 0.0 or time_a <= 0.0 or time_b <= 0.0:
                raise ValueError("Density and times must be positive.")

            # covariance = basis * sqrt(D)
            basis = (
                math.sqrt(time_a) + math.sqrt(time_b) - math.sqrt(abs(time_a - time_b))
            ) / (density * math.sqrt(2.0 * math.pi))
            weight = 1.0 / (sigma * sigma)
            numerator_terms.append(weight * basis * observed)
            denominator_terms.append(weight * basis * basis)

        sqrt_diffusivity = math.fsum(numerator_terms) / math.fsum(denominator_terms)
        sqrt_diffusivity = min(
            math.sqrt(_MAX_DIFFUSIVITY),
            max(math.sqrt(_MIN_DIFFUSIVITY), sqrt_diffusivity),
        )
        self.diffusivity = sqrt_diffusivity * sqrt_diffusivity
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise ValueError("Call fit(records) before predict(experiments).")
        return predict_at(experiments, self.diffusivity)
