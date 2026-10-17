import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time covariance coefficient for the tagged particle.

    The initially equally-spaced file is a *quenched* initial condition.  Its
    covariance coefficient is therefore the quenched single-file result,
    rather than the corresponding Poisson/annealed result.  In particular,
    its unequal-time dependence is through ``sqrt(time_a + time_b)``.
    """
    return (
        np.sqrt(float(diffusivity) / np.pi)
        / float(density)
        * (
            np.sqrt(float(time_a) + float(time_b))
            - np.sqrt(abs(float(time_a) - float(time_b)))
        )
    )


def predict_at(experiments, diffusivity):
    return np.asarray([
        covariance(float(diffusivity), float(e["density"]),
                   float(e["time_a"]), float(e["time_b"]))
        for e in experiments
    ], dtype=float)


class Model:
    """One-parameter model for the tagged-particle covariance experiment."""

    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Estimate the shared diffusivity by weighted Gaussian least squares.

        For each record the prediction is ``sqrt(D) * q`` for a known q.
        Fitting ``sqrt(D)`` consequently has a closed-form weighted least
        squares solution.  The optimization is constrained to the stated
        physical interval for D.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        factors = np.asarray(
            [
                covariance(
                    1.0,
                    record["input"]["density"],
                    record["input"]["time_a"],
                    record["input"]["time_b"],
                )
                for record in records
            ],
            dtype=float,
        )
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if (
            not np.isfinite(factors).all()
            or not np.isfinite(values).all()
            or not np.isfinite(sigmas).all()
            or np.any(sigmas <= 0.0)
        ):
            raise ValueError("calibration records must contain finite values and positive sigma")

        weights = 1.0 / np.square(sigmas)
        amplitude = np.sum(weights * factors * values) / np.sum(weights * np.square(factors))
        # D is known to be in [0.8, 1.2], hence sqrt(D) is positive and bounded.
        amplitude = np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(np.square(amplitude))
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.diffusivity)
