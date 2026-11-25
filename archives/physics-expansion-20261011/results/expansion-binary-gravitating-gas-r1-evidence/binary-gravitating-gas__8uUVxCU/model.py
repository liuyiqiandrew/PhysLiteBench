"""Critical temperature of a self-gravitating, two-species isothermal gas."""

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


@lru_cache(maxsize=1)
def single_species_coefficient():
    """Return the dimensionless canonical instability coefficient.

    For one species the regular isothermal Lane--Emden equation is

        psi'' + 2 psi'/x = exp(-psi).

    The first canonical turning point is the zero of
    ``x exp(-psi) - psi'``.  Keeping this separate also makes the limiting
    cases q=0 and chi=1 both fast and numerically exact.
    """

    start = 1e-5

    def rhs(x, y):
        return [y[1], np.exp(-y[0]) - 2.0 * y[1] / x]

    profile = solve_ivp(
        rhs,
        (start, 12.0),
        [start**2 / 6.0, start / 3.0],
        rtol=2e-11,
        atol=2e-13,
        dense_output=True,
    ).sol

    def condition(alpha):
        psi, gradient = profile(alpha)
        return alpha * np.exp(-psi) - gradient

    alpha = brentq(condition, 5.0, 12.0)
    return float(1.0 / (alpha * profile(alpha)[1]))


def _two_species_profile(alpha, central_heavy_fraction, mass_ratio):
    """Integrate a dimensionless two-species equilibrium profile.

    The scale is set by the total central mass density, so the source at
    the origin is one even when the heavy species is strongly concentrated.
    The two final integrals are unweighted volume integrals; the central
    species fraction supplies their mass-density weights.
    """

    start = 1e-5
    p = central_heavy_fraction
    chi = mass_ratio

    def rhs(x, y):
        psi, gradient, light_integral, heavy_integral = y
        light = np.exp(-psi)
        heavy = np.exp(-chi * psi)
        source = (1.0 - p) * light + p * heavy
        return [
            gradient,
            source - 2.0 * gradient / x,
            light * x * x,
            heavy * x * x,
        ]

    solution = solve_ivp(
        rhs,
        (start, alpha),
        [start**2 / 6.0, start / 3.0, start**3 / 3.0, start**3 / 3.0],
        method="DOP853",
        rtol=3e-10,
        atol=2e-12,
    )
    return solution.y[:, -1]


def _coefficient_at_alpha(alpha, mass_ratio, heavy_fraction):
    """Get theta*R/(G*M) at a specified dimensionless wall radius."""

    q = heavy_fraction
    chi = mass_ratio

    def mass_fraction(p):
        _, _, light_integral, heavy_integral = _two_species_profile(
            alpha, p, chi
        )
        total = (1.0 - p) * light_integral + p * heavy_integral
        return p * heavy_integral / total

    # Heavy particles are more centrally concentrated, so their central
    # fraction p is at least their global mass fraction q.
    p = brentq(
        lambda value: mass_fraction(value) - q,
        q,
        1.0 - 1e-12,
        xtol=2e-12,
        rtol=1e-11,
    )
    _, _, light_integral, heavy_integral = _two_species_profile(alpha, p, chi)
    total_integral = (1.0 - p) * light_integral + p * heavy_integral
    return float(alpha / total_integral)


@lru_cache(maxsize=512)
def _two_species_coefficient(mass_ratio, heavy_fraction):
    """Find the first canonical instability on the fixed-composition branch."""

    chi = float(mass_ratio)
    q = float(heavy_fraction)
    if q == 0.0 or chi == 1.0:
        return single_species_coefficient()

    # The branch starts at infinite temperature as alpha -> 0 and has its
    # first minimum between these bounds throughout chi <= 5, q <= 1/2. The
    # search is in log(alpha), resolving the dilute and concentrated ends
    # evenly.
    result = minimize_scalar(
        lambda log_alpha: _coefficient_at_alpha(float(np.exp(log_alpha)), chi, q),
        bounds=(np.log(0.5), np.log(20.0)),
        method="bounded",
        options={"xatol": 2e-7},
    )
    return float(result.fun)


def coefficient(ratio, fraction):
    """Return the dimensionless first-instability coefficient."""

    return _two_species_coefficient(float(ratio), float(fraction))


def predict_at(experiments, gravity):
    return np.asarray(
        [
            gravity
            * float(experiment["mass"])
            / float(experiment["radius"])
            * coefficient(
                float(experiment["mass_ratio"]),
                float(experiment["heavy_fraction"]),
            )
            for experiment in experiments
        ],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.gravity = None

    def fit(self, records):
        """Fit the shared gravitational strength by weighted least squares."""

        features = []
        values = []
        weights = []
        for record in records:
            experiment = record["input"]
            features.append(
                float(experiment["mass"])
                / float(experiment["radius"])
                * coefficient(
                    float(experiment["mass_ratio"]),
                    float(experiment["heavy_fraction"]),
                )
            )
            values.append(float(record["value"]))
            sigma = float(record["sigma"])
            weights.append(1.0 / (sigma * sigma))

        x = np.asarray(features, dtype=float)
        y = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        fitted = np.sum(w * x * y) / np.sum(w * x * x)
        self.gravity = float(np.clip(fitted, 0.7, 1.4))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.gravity)
