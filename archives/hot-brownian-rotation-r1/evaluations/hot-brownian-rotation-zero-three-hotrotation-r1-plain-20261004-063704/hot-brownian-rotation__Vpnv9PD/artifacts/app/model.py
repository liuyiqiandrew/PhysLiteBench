"""Spectrum model for the fluctuating torque on a heated sphere.

The rotational unsteady-Stokes solution is used as the adjoint field for the
random stress.  Since the temperature is not uniform, the spectrum is the
local stress variance weighted by the squared strain of that solution.
"""

import numpy as np


try:  # scipy is available in the execution environment, but keep a fallback.
    from scipy.special import expn as _expn
except ImportError:  # pragma: no cover - exercised only in minimal environments
    _expn = None


def impedance(e, viscosity):
    """Rotational hydrodynamic impedance of a sphere."""

    a = float(e["radius"])
    omega = float(e["omega"])
    viscosity = float(viscosity)
    z = a * np.sqrt(-1j * omega / viscosity)
    return 8 * np.pi * viscosity * a**3 * (1 + z * z / (3 * (1 + z)))


def _exponential_moments(lam):
    """Return J_m = integral exp(-lam*(x-1))/x**m dx from x=1 to infinity."""

    if lam <= 0:
        return np.full(5, np.inf, dtype=float)

    if _expn is not None:
        return np.exp(lam) * np.asarray([_expn(m, lam) for m in range(1, 6)])

    # Self-contained fallback for minimal Python installations.  The normal
    # environment uses scipy.special.expn above.
    if lam <= 1.0:
        term = 1.0
        series = 0.0
        for k in range(1, 200):
            term *= -lam / k
            add = term / k
            series += add
            if abs(add) < 2e-16 * max(1.0, abs(series)):
                break
        j = np.exp(lam) * (-0.5772156649015329 - np.log(lam) - series)
    else:
        # E1(lam)*exp(lam) = integral_0^infinity exp(-t)/(lam+t) dt.
        # Gauss-Laguerre quadrature avoids a dependency on a special-function
        # library in this fallback and is very accurate for lam >= 1.
        nodes, weights = np.polynomial.laguerre.laggauss(64)
        j = float(np.dot(weights, 1.0 / (lam + nodes)))

    moments = [j]
    for m in range(1, 5):
        moments.append((1.0 - lam * moments[-1]) / m)
    return np.asarray(moments)


def _strain_integrals(z):
    """Return radial strain integrals for ambient and 1/r temperature terms."""

    if z == 0:
        return 3.0, 2.25

    lam = 2.0 * float(np.real(z))
    # |3 + 3 z x + z^2 x^2|^2 = sum(d[k] x^k).
    d = np.asarray(
        [
            9.0,
            18.0 * np.real(z),
            9.0 * abs(z) ** 2 + 6.0 * np.real(z * z),
            6.0 * abs(z) ** 2 * np.real(z),
            abs(z) ** 4,
        ],
        dtype=float,
    )
    moments = _exponential_moments(lam)

    # The x^4 and x^3 factors cancel four and three powers of x in the
    # derivative denominator, respectively.  The x^0 moment is 1/lam.
    j0 = np.asarray([moments[3 - k] if k < 4 else 1.0 / lam for k in range(5)])
    j1 = moments[4::-1]
    scale = abs(1.0 + z) ** 2
    return float(np.dot(d, j0) / scale), float(np.dot(d, j1) / scale)


def spectrum(e, viscosity):
    """Return the two-sided connected torque spectrum for one experiment."""

    a = float(e["radius"])
    ambient = float(e["ambient"])
    rise = float(e["rise"])
    omega = float(e["omega"])
    viscosity = float(viscosity)

    z = a * np.sqrt(-1j * omega / viscosity)
    i_ambient, i_rise = _strain_integrals(z)
    # 4*viscosity times the squared strain integral, including its angular
    # factor, gives this prefactor.
    value = (16.0 * np.pi * viscosity * a**3 / 3.0) * (
        ambient * i_ambient + rise * i_rise
    )
    return float(value)


def predict_at(experiments, viscosity):
    return np.asarray([spectrum(e, viscosity) for e in experiments], dtype=float)


def _weighted_error(viscosity, inputs, values, sigmas):
    residual = (predict_at(inputs, viscosity) - values) / sigmas
    return float(np.dot(residual, residual))


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        inputs = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record.get("sigma", 0.02) for record in records], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
            raise ValueError("calibration values and uncertainties must be finite")

        # At zero frequency the model is exactly linear in viscosity.  This is
        # the fast and statistically exact fit for the supplied calibration.
        if all(float(e["omega"]) == 0.0 for e in inputs):
            a = np.asarray([float(e["radius"]) for e in inputs])
            ambient = np.asarray([float(e["ambient"]) for e in inputs])
            rise = np.asarray([float(e["rise"]) for e in inputs])
            design = 16.0 * np.pi * a**3 * (ambient + 0.75 * rise)
            weights = 1.0 / sigmas**2
            viscosity = np.sum(weights * design * values) / np.sum(weights * design**2)
            self.viscosity = float(np.clip(viscosity, 0.7, 1.4))
            return self

        # A bounded one-dimensional golden-section search also handles records
        # containing nonzero frequencies.
        lo, hi = 0.7, 1.4
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        c = hi - (hi - lo) / phi
        d = lo + (hi - lo) / phi
        fc = _weighted_error(c, inputs, values, sigmas)
        fd = _weighted_error(d, inputs, values, sigmas)
        for _ in range(80):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - (hi - lo) / phi
                fc = _weighted_error(c, inputs, values, sigmas)
            else:
                lo, c, fc = c, d, fd
                d = lo + (hi - lo) / phi
                fd = _weighted_error(d, inputs, values, sigmas)
        self.viscosity = float((lo + hi) / 2.0)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.viscosity)
