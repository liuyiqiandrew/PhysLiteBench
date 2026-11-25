"""Large-deviation model for heat in the circulating harmonic trap."""

import numpy as np


def response(experiment):
    """Return the heat rate function for unit mobility.

    Time can be rescaled by ``mobility * stiffness``.  In those units the
    scaled cumulant generating function of heat is

        1 - sqrt(1 + 4 r**2 s (1 - s)).

    The usual Legendre branch is not the full heat rate function: the
    conservative part of heat is minus the (unconditioned) change in trap
    energy.  Its exponential stationary tails give the two linear pieces
    below.  They matter for the outer part of the advertised z range.
    """
    stiffness = float(experiment["stiffness"])
    drive_ratio = float(experiment["drive_ratio"])
    z = float(experiment["relative_heat"])
    a = drive_ratio * drive_ratio

    # A large final trap energy makes sufficiently negative heat.  The
    # stationary energy tail has rate -q, where q = 2*a*z in scaled time.
    if z <= -1.0:
        dimensionless_rate = -2.0 * a * z
    else:
        # For weak circulation, a large initial energy creates a second
        # boundary branch.  It exists only for a < 1/3.
        upper_z = (3.0 - a) / (1.0 - 3.0 * a) if a < (1.0 / 3.0) else np.inf
        if z > upper_z:
            dimensionless_rate = (
                2.0 * a * (1.0 - a) * z - 4.0 * a
            ) / (1.0 + a)
        else:
            dimensionless_rate = (
                np.sqrt((1.0 + a) * (1.0 + a * z * z)) - a * z - 1.0
            )

    return stiffness * dimensionless_rate


def predict_at(experiments, mobility):
    """Evaluate the rate function for a common mobility."""
    return float(mobility) * np.asarray(
        [response(experiment) for experiment in experiments], dtype=float
    )


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by the Gaussian maximum-likelihood rule."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        unit_mobility_rates = np.asarray(
            [response(record["input"]) for record in records], dtype=float
        )
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)

        if (
            not np.isfinite(unit_mobility_rates).all()
            or not np.isfinite(values).all()
            or not np.isfinite(sigmas).all()
            or np.any(sigmas <= 0.0)
        ):
            raise ValueError("calibration values and positive sigmas must be finite")

        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * np.square(unit_mobility_rates))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("calibration records do not identify mobility")

        estimate = np.sum(weights * unit_mobility_rates * values) / denominator
        # The apparatus specifies the physically admissible parameter range,
        # so this is the constrained Gaussian maximum-likelihood estimate.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.mobility)
