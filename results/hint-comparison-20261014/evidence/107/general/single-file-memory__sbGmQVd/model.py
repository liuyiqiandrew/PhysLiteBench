import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time tagged-particle covariance for the stated lattice start.

    The initial configuration in the problem is fixed rather than sampled from
    equilibrium.  This is the quenched single-file result; in particular its
    two-time dependence is different from the often-used equilibrium result
    containing ``sqrt(time_a + time_b)``.
    """
    return (
        np.sqrt(diffusivity / (2.0 * np.pi))
        / density
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
        """Infer the common diffusivity by Gaussian weighted least squares.

        For each record the prediction is ``sqrt(D) * c_i``.  Thus fitting
        ``sqrt(D)`` is a one-parameter weighted linear least-squares problem,
        with a closed-form solution.  The physical prior range documented for
        the apparatus is imposed after the fit.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        controls = np.asarray(
            [
                (float(record["input"]["density"]),
                 float(record["input"]["time_a"]),
                 float(record["input"]["time_b"]))
                for record in records
            ],
            dtype=float,
        )
        observed = np.asarray([float(record["value"]) for record in records], dtype=float)
        sigma = np.asarray([float(record["sigma"]) for record in records], dtype=float)

        if (not np.isfinite(controls).all() or not np.isfinite(observed).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0.0)):
            raise ValueError("calibration records must contain finite values and positive sigma")

        # ``covariance(1, ...)`` is the coefficient multiplying sqrt(D).
        coefficients = np.asarray(
            [covariance(1.0, density, time_a, time_b)
             for density, time_a, time_b in controls],
            dtype=float,
        )
        weights = 1.0 / np.square(sigma)
        amplitude = np.sum(weights * coefficients * observed) / np.sum(
            weights * np.square(coefficients)
        )

        # D is known to lie in [0.8, 1.2], equivalently for its non-negative
        # square-root amplitude.  Clipping is the constrained MLE.
        amplitude = np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(amplitude * amplitude)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.diffusivity)
