import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Return the limiting covariance coefficient for one experiment.

    For a single-file system the long-time tagged-particle covariance is
    proportional to the square root of the isolated-particle diffusivity.
    Writing the result this way also makes the parameterization used by
    :meth:`Model.fit` explicit.
    """
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)

    return (
        np.sqrt(diffusivity / (2 * np.pi)) / density
        * (np.sqrt(time_a) + np.sqrt(time_b) - np.sqrt(abs(time_a - time_b)))
    )


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
        """Estimate the common diffusivity from calibration records.

        The prediction has the form ``y_i = a_i * sqrt(D)``, where ``a_i``
        depends only on the controls in record ``i``.  Consequently the
        weighted Gaussian least-squares estimate has a closed form.  The
        known diffusivity range is applied in the fitted parameter
        ``sqrt(D)`` so that the estimate remains physically admissible.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

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

        coefficients = np.asarray(coefficients, dtype=float)
        values = np.asarray(values, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        if (
            not np.isfinite(coefficients).all()
            or not np.isfinite(values).all()
            or not np.isfinite(sigmas).all()
            or np.any(sigmas <= 0)
            or np.all(coefficients == 0)
        ):
            raise ValueError("calibration records must contain finite, usable values")

        weights = 1.0 / np.square(sigmas)
        sqrt_diffusivity = np.sum(weights * coefficients * values) / np.sum(
            weights * np.square(coefficients)
        )
        sqrt_diffusivity = np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(np.square(sqrt_diffusivity))
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise ValueError("fit the model before predicting")
        return predict_at(experiments, self.diffusivity)
