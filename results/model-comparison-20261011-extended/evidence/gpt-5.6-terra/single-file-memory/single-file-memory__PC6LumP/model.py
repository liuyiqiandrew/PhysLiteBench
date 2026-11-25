import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time tagged-particle covariance for the ordered Brownian file.

    The initial configuration in the problem is deterministic.  Consequently
    this is the quenched single-file result (as opposed to the larger annealed
    result, which would include fluctuations of the initial particle count).
    """
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)
    if not all(np.isfinite(x) for x in (diffusivity, density, time_a, time_b)):
        raise ValueError("diffusivity, density, and times must be finite")
    if diffusivity < 0 or density <= 0 or time_a < 0 or time_b < 0:
        raise ValueError("diffusivity and times must be non-negative, and density positive")

    time_factor = np.sqrt(time_a) + np.sqrt(time_b) - np.sqrt(abs(time_a - time_b))
    return np.sqrt(diffusivity / (2.0 * np.pi)) * time_factor / density


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
        """Estimate the shared diffusivity by weighted Gaussian least squares.

        Since every model value is ``sqrt(D)`` times a known coefficient,
        fitting that parameter has a closed-form weighted least-squares
        solution.  It is then clipped to the documented physical range.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        coefficients = []
        observations = []
        weights = []
        for record in records:
            try:
                experiment = record["input"]
                observation = float(record["value"])
                sigma = float(record["sigma"])
                coefficient = covariance(
                    1.0,
                    experiment["density"],
                    experiment["time_a"],
                    experiment["time_b"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("each calibration record must contain valid input, value, and sigma") from exc

            if not np.isfinite(observation) or not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("calibration values must be finite and sigma must be positive")
            coefficients.append(coefficient)
            observations.append(observation)
            weights.append(1.0 / sigma**2)

        coefficients = np.asarray(coefficients, dtype=float)
        observations = np.asarray(observations, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = np.sum(weights * coefficients**2)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("calibration inputs provide no information about diffusivity")

        sqrt_diffusivity = np.sum(weights * coefficients * observations) / denominator
        sqrt_diffusivity = np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(sqrt_diffusivity**2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.diffusivity)
