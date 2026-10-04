"""Spectrum of the fluctuating torque on a heated, fixed sphere.

The deterministic rotational flow is the l=1 toroidal solution of the
linearized unsteady Stokes equation.  By reciprocity, a local thermal stress
contributes in proportion to the local viscous dissipation density.  This is
why the radial temperature profile produces a frequency-dependent effective
temperature.
"""

import numpy as np


def _z(e, viscosity):
    """Dimensionless inverse penetration depth, with the stated Fourier sign."""
    return e["radius"] * np.sqrt(-1j * e["omega"] / viscosity)


def impedance(e, viscosity):
    """Complex rotational hydrodynamic impedance of the sphere."""
    a = float(e["radius"])
    z = _z(e, viscosity)
    return 8.0 * np.pi * viscosity * a**3 * (
        1.0 + z * z / (3.0 * (1.0 + z))
    )


def _inverse_power_moments(p):
    """Return J_m = integral exp(-p*(x-1))*x**(-m) dx, m=0,...,4."""
    if p == 0.0:
        return np.array([np.inf, 1.0, 0.5, 1.0 / 3.0, 0.25])

    # J_1 = exp(p) E1(p).  scipy is convenient when present; the fallback is
    # included so the model still works with only numpy available.
    try:
        from scipy.special import exp1

        j1 = float(np.exp(p) * exp1(p))
    except ImportError:  # pragma: no cover
        if p < 1.0:
            gamma = 0.5772156649015329
            term = -p
            series = term
            for k in range(2, 80):
                term *= -p / k
                series += term / k
            j1 = np.exp(p) * (-gamma - np.log(p) - series)
        else:
            # Continued fraction for E1(p), evaluated by modified Lentz's
            # method.  This is well behaved for p >= 1, unlike an asymptotic
            # series truncated at a fixed number of terms.
            tiny = np.finfo(float).tiny
            b = p + 1.0
            c = 1.0 / tiny
            d = 1.0 / b
            h = d
            for k in range(1, 200):
                a = -float(k * k)
                b += 2.0
                d = 1.0 / (a * d + b)
                c = b + a / c
                delta = c * d
                h *= delta
                if abs(delta - 1.0) < 1e-15:
                    break
            j1 = h

    moments = np.empty(5, dtype=float)
    moments[0] = 1.0 / p
    moments[1] = j1
    for m in range(1, 4):
        moments[m + 1] = (1.0 - p * moments[m]) / m
    return moments


def _temperature_weight(e, viscosity):
    """Dissipation-weighted average of ``a/r`` at the given frequency."""
    z = _z(e, viscosity)
    if z == 0:
        return 0.75

    p = 2.0 * float(np.real(z))
    moments = _inverse_power_moments(p)

    # |z^2 + 3z/x + 3/x^2|^2, grouped by powers x^0,...,x^-4.
    coefficients = np.array(
        [
            abs(z) ** 4,
            6.0 * abs(z) ** 2 * np.real(z),
            9.0 * abs(z) ** 2 + 6.0 * np.real(z * z),
            18.0 * np.real(z),
            9.0,
        ],
        dtype=float,
    )
    denominator = coefficients[0] * moments[0] + np.dot(
        coefficients[1:], moments[1:]
    )

    # The numerator has one extra factor a/r = 1/x.  J_5 is obtained from
    # the same integration-by-parts recurrence as J_1,...,J_4.
    j5 = (1.0 - p * moments[4]) / 4.0
    numerator = np.dot(coefficients, [moments[1], moments[2], moments[3], moments[4], j5])
    return float(numerator / denominator)


def spectrum(e, viscosity):
    """Two-sided connected torque spectrum for one experiment."""
    ambient = float(e["ambient"])
    rise = float(e["rise"])
    real_impedance = float(np.real(impedance(e, viscosity)))
    temperature = ambient + rise * _temperature_weight(e, viscosity)
    value = 2.0 * real_impedance * temperature
    return float(value) if np.isfinite(value) else np.nan


def predict_at(experiments, viscosity):
    """Evaluate spectra in input order and return a 1-D float array."""
    return np.asarray([spectrum(e, viscosity) for e in experiments], dtype=float)


def _weighted_loss(records, viscosity):
    prediction = predict_at((record["input"] for record in records), viscosity)
    observed = np.asarray([record["value"] for record in records], dtype=float)
    sigma = np.asarray([record.get("sigma", 0.02) for record in records], dtype=float)
    residual = (prediction - observed) / sigma
    return float(np.dot(residual, residual))


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit viscosity by weighted least squares, constrained to [0.7, 1.4]."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        # At zero frequency the model is linear in viscosity, so use the
        # exact weighted least-squares estimate for the supplied calibration.
        if all(float(r["input"].get("omega", 0.0)) == 0.0 for r in records):
            x = np.asarray(
                [
                    16.0
                    * np.pi
                    * float(r["input"]["radius"]) ** 3
                    * (
                        float(r["input"]["ambient"])
                        + 0.75 * float(r["input"]["rise"])
                    )
                    for r in records
                ]
            )
            y = np.asarray([float(r["value"]) for r in records])
            w = np.asarray([1.0 / float(r.get("sigma", 0.02)) ** 2 for r in records])
            estimate = float(np.sum(w * x * y) / np.sum(w * x * x))
        else:
            # Bounded golden-section search also supports non-DC calibration
            # records without requiring an optimization dependency.
            lo, hi = 0.7, 1.4
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            x1 = hi - (hi - lo) / phi
            x2 = lo + (hi - lo) / phi
            f1, f2 = _weighted_loss(records, x1), _weighted_loss(records, x2)
            for _ in range(100):
                if f1 > f2:
                    lo, x1, f1 = x1, x2, f2
                    x2 = lo + (hi - lo) / phi
                    f2 = _weighted_loss(records, x2)
                else:
                    hi, x2, f2 = x2, x1, f1
                    x1 = hi - (hi - lo) / phi
                    f1 = _weighted_loss(records, x1)
            estimate = 0.5 * (lo + hi)

        self.viscosity = float(np.clip(estimate, 0.7, 1.4))
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.viscosity)
