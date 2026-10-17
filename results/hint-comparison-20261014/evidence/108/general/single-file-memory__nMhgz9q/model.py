import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Return the long-time covariance coefficient for one experiment.

    For a single-file system the tagged-particle covariance is

        sqrt(D / (2*pi)) / density
        * (sqrt(t_a) + sqrt(t_b) - sqrt(abs(t_a - t_b))).

    In particular, the coefficient is proportional to ``sqrt(D)``.  Keeping
    that dependence explicit is useful in ``Model.fit`` below.
    """
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)
    return (np.sqrt(diffusivity / (2.0 * np.pi)) / density
            * (np.sqrt(time_a) + np.sqrt(time_b)
               - np.sqrt(abs(time_a - time_b))))


def predict_at(experiments, diffusivity):
    if diffusivity is None:
        raise ValueError("Model must be fitted before predict() can be used")
    return np.asarray([
        covariance(diffusivity, e["density"], e["time_a"], e["time_b"])
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Infer the common diffusivity from calibration observations.

        Since ``covariance(D, ...) = sqrt(D) * covariance(1, ...)``, fitting
        ``sqrt(D)`` is a one-parameter weighted linear least-squares problem.
        This is also the maximum-likelihood estimate for the stated Gaussian
        instrument errors.  The documented physical bounds are applied to the
        fitted parameter so that predictions remain physically meaningful.
        """
        rows = list(records)
        if not rows:
            raise ValueError("At least one calibration record is required")

        design = []
        observations = []
        weights = []
        for record in rows:
            experiment = record["input"]
            observation = float(record["value"])
            sigma = float(record["sigma"])
            if not np.isfinite(observation):
                raise ValueError("Calibration values must be finite")
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("Calibration sigmas must be positive and finite")

            design.append(covariance(1.0, experiment["density"],
                                     experiment["time_a"],
                                     experiment["time_b"]))
            observations.append(observation)
            weights.append(1.0 / sigma**2)

        design = np.asarray(design, dtype=float)
        observations = np.asarray(observations, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = np.sum(weights * design * design)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("Calibration records do not define a usable fit")

        sqrt_diffusivity = np.sum(weights * design * observations) / denominator
        sqrt_diffusivity = np.clip(sqrt_diffusivity,
                                   np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(sqrt_diffusivity**2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
