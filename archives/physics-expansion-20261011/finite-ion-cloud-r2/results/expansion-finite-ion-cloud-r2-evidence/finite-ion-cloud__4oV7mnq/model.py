"""Finite-cell osmotic coefficient for the charged ionic disk."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import hyp2f1


def _one_ion_coefficient(gamma, ratio):
    """Return phi for one ion, using its radial partition function."""
    length = np.log(ratio)
    exponent = 2.0 - gamma

    # d log(Z)/d length, written this way to remain stable when exponent is
    # close to zero. The area derivative contributes (1-exp(-2 length))/2.
    if abs(exponent * length) < 1.0e-8:
        logarithmic_slope = 1.0 / length
    else:
        logarithmic_slope = exponent / (-np.expm1(-exponent * length))
    return -np.expm1(-2.0 * length) * logarithmic_slope / 2.0


def _two_ion_coefficient(gamma, ratio):
    """Return phi for two ions from the finite canonical partition function.

    Ordering the two radii reduces the four-dimensional configurational
    integral to a one-dimensional integral. The angular integral is

        2 pi * hyp2f1(-gamma/2, -gamma/2; 1; exp(-2 s)),

    where s is the difference of the ordered log-radii. Differentiating the
    resulting triangular domain with respect to its upper radius gives the
    pressure without a finite-difference approximation.
    """
    length = np.log(ratio)
    radial_inner = 2.0 - 2.0 * gamma
    radial_outer = 2.0 - gamma
    total_radial = radial_inner + radial_outer

    def angular_average(separation):
        return hyp2f1(
            -0.5 * gamma,
            -0.5 * gamma,
            1.0,
            np.exp(-2.0 * separation),
        )

    def triangular_width(separation):
        scaled = total_radial * (length - separation)
        if abs(total_radial) < 1.0e-10:
            return length - separation
        return np.expm1(scaled) / total_radial

    partition_integral = quad(
        lambda separation: (
            np.exp(radial_outer * separation)
            * angular_average(separation)
            * triangular_width(separation)
        ),
        0.0,
        length,
        epsabs=2.0e-11,
        epsrel=2.0e-11,
    )[0]

    boundary_integral = np.exp(total_radial * length) * quad(
        lambda separation: (
            np.exp(-radial_inner * separation) * angular_average(separation)
        ),
        0.0,
        length,
        epsabs=2.0e-11,
        epsrel=2.0e-11,
    )[0]

    # A/(2 pi R^2) = (1-exp(-2 length))/2, and phi has one further
    # factor 1/N for the two-ion density.
    logarithmic_slope = boundary_integral / partition_integral
    return -np.expm1(-2.0 * length) * logarithmic_slope / 4.0


def coefficient(gamma, count, ratio):
    """Finite-cell osmotic coefficient at coupling ``gamma``."""
    if count == 1:
        return _one_ion_coefficient(gamma, ratio)
    if count == 2:
        return _two_ion_coefficient(gamma, ratio)
    raise ValueError("ion_count must be 1 or 2")


def predict_at(experiments, coupling):
    return np.asarray(
        [
            coefficient(
                experiment["temperature_multiplier"] * coupling,
                experiment["ion_count"],
                experiment["radius_ratio"],
            )
            for experiment in experiments
        ],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        """Fit the shared reference coupling to the noisy one-ion records."""
        records = list(records)

        def objective(coupling):
            residuals = []
            for record in records:
                experiment = record["input"]
                prediction = coefficient(
                    experiment["temperature_multiplier"] * coupling,
                    experiment["ion_count"],
                    experiment["radius_ratio"],
                )
                residuals.append(
                    (prediction - record["value"]) / record["sigma"]
                )
            return float(np.dot(residuals, residuals))

        if not records:
            raise ValueError("at least one calibration record is required")

        result = minimize_scalar(
            objective,
            bounds=(0.7, 1.4),
            method="bounded",
            options={"xatol": 1.0e-12},
        )
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        if self.coupling is None:
            raise ValueError('fit the model before predicting')
        return predict_at(experiments,self.coupling)
