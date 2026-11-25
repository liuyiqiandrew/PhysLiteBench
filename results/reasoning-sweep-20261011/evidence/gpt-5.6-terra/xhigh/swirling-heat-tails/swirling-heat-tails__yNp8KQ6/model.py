"""Large-deviation model for the heat in the circulating harmonic trap."""

import numpy as np


# The physical range supplied for the (common) apparatus mobility.
_MOBILITY_LIMITS = (0.8, 1.3)


def response(experiment):
    """Return the rate function apart from its mobility factor.

    For a preparation with drive ratio ``r``, the stationary heat rate is
    ``q_bar = 2 * mobility * stiffness * r**2``.  The central part of the
    normalized heat rate function is

        sqrt((1+r**2)*(1+r**2*z**2)) - 1 - r**2*z.

    Heat also contains the unconditioned endpoint-energy change.  Its
    exponential tails restrict the heat SCGF to ``lambda > -1`` and, for
    ``r**2 < 1/3``, to an additional upper bound.  Legendre transformation
    therefore gives linear endpoint-dominated branches outside the central
    region.  This matters in the requested full relative-heat domain.
    """
    stiffness = float(experiment["stiffness"])
    drive_ratio = float(experiment["drive_ratio"])
    relative_heat = float(experiment["relative_heat"])

    drive_squared = drive_ratio * drive_ratio
    radical = np.sqrt((1.0 + drive_squared) *
                      (1.0 + drive_squared * relative_heat * relative_heat))
    if relative_heat <= -1.0:
        # Final high-energy endpoint: lambda reaches its lower bound -1.
        dimensionless_rate = -2.0 * drive_squared * relative_heat
    elif drive_squared < 1.0 / 3.0:
        upper_relative_heat = (3.0 - drive_squared) / (1.0 - 3.0 * drive_squared)
        if relative_heat >= upper_relative_heat:
            # Initial high-energy endpoint at the upper SCGF bound.
            dimensionless_rate = (
                2.0 * drive_squared
                * ((1.0 - drive_squared) * relative_heat - 2.0)
                / (1.0 + drive_squared)
            )
        else:
            # Rationalized central branch, stable close to z=1.
            dimensionless_rate = (
                drive_squared * (relative_heat - 1.0) ** 2
                / (radical + 1.0 + drive_squared * relative_heat)
            )
    else:
        # Rationalized central branch, stable close to z=1.
        dimensionless_rate = (
            drive_squared * (relative_heat - 1.0) ** 2
            / (radical + 1.0 + drive_squared * relative_heat)
        )
    return stiffness * dimensionless_rate


def predict_at(experiments, mobility):
    """Evaluate the physical rate function for a common mobility."""
    return float(mobility) * np.asarray(
        [response(experiment) for experiment in experiments], dtype=float
    )


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by the known-variance Gaussian MLE."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        unit_mobility_rates = np.asarray(
            [response(record["input"]) for record in records], dtype=float
        )
        observed_rates = np.asarray(
            [record["value"] for record in records], dtype=float
        )
        standard_deviations = np.asarray(
            [record["sigma"] for record in records], dtype=float
        )

        if (not np.isfinite(unit_mobility_rates).all()
                or not np.isfinite(observed_rates).all()
                or not np.isfinite(standard_deviations).all()
                or np.any(standard_deviations <= 0.0)):
            raise ValueError("calibration rates and standard deviations must be finite; sigma must be positive")

        weights = 1.0 / standard_deviations ** 2
        denominator = np.sum(weights * unit_mobility_rates ** 2)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("calibration records do not identify mobility")

        unconstrained_mle = np.sum(weights * unit_mobility_rates * observed_rates) / denominator
        self.mobility = float(np.clip(unconstrained_mle, *_MOBILITY_LIMITS))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.mobility)
