import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Return the limiting two-time covariance coefficient.

    In the long-time limit, the marked displacement is determined by the net
    number of independent Brownian paths crossing its initial position.  The
    crossing indicators at two times on the same path are correlated.  After
    summing their covariance over the initially fixed particle positions, the
    coefficient is the expression below.
    """
    d = float(diffusivity)
    rho = float(density)
    ta = float(time_a)
    tb = float(time_b)

    return (
        np.sqrt(d / np.pi) / rho
        * (np.sqrt(ta + tb) - np.sqrt(abs(ta - tb)))
    )


def predict_at(experiments, diffusivity):
    """Evaluate the covariance model in the input order."""
    return np.asarray(
        [
            covariance(
                diffusivity,
                float(experiment["density"]),
                float(experiment["time_a"]),
                float(experiment["time_b"]),
            )
            for experiment in experiments
        ],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Fit the common diffusivity by weighted least squares.

        The covariance model is linear in sqrt(diffusivity), so the weighted
        maximum-likelihood estimate has a closed form.
        """
        coefficients = []
        values = []
        sigmas = []

        for record in records:
            experiment = record["input"]
            coefficients.append(
                covariance(
                    1.0,
                    experiment["density"],
                    experiment["time_a"],
                    experiment["time_b"],
                )
            )
            values.append(float(record["value"]))
            sigmas.append(float(record["sigma"]))

        if not coefficients:
            raise ValueError("at least one calibration record is required")

        coefficients = np.asarray(coefficients, dtype=float)
        values = np.asarray(values, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)

        if (
            not np.isfinite(coefficients).all()
            or not np.isfinite(values).all()
            or not np.isfinite(sigmas).all()
            or np.any(sigmas <= 0.0)
            or np.any(coefficients <= 0.0)
        ):
            raise ValueError("calibration records must contain finite valid values")

        weights = 1.0 / np.square(sigmas)
        sqrt_diffusivity = np.sum(weights * coefficients * values) / np.sum(
            weights * np.square(coefficients)
        )
        # The calibration is specified to come from this physical interval.
        sqrt_diffusivity = np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(np.square(sqrt_diffusivity))
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("fit the model before requesting predictions")
        return predict_at(experiments, self.diffusivity)
