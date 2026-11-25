"""Thermodynamic current SCGF of the half-filled weakly asymmetric ring.

The diffusivity multiplies the entire microscopic generator, so it also
multiplies the SCGF without changing the location of the phase transition.
"""
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk, elliprf, elliprj


@lru_cache(maxsize=4096)
def _unit_scgf(field, bias):
    """SCGF for D=1, including the symmetry-broken density profile.

    In macroscopic fluctuation theory, at half filling the optimal wave has
    zero velocity. Writing m=2*rho-1 and k=field+bias, its variational value is

      k^2/(4 <1/(1-m^2)>) - field^2 <1-m^2>/4
          - <m'^2/(1-m^2)>/4.

    The uniform solution loses stability at field^2-k^2=4*pi^2. Beyond
    this point m=a*sn(field*b*x, q), q=a^2/b^2, with period one. The
    period and counting constraint determine b and q through complete
    elliptic integrals. This evaluates the continuum limit directly.
    """
    flat = bias * (bias + 2.0 * field) / 4.0
    if -4.0 * flat <= 4.0 * np.pi**2:
        return flat

    k = abs(field + bias)
    # At k=0 the outer turning point is b=1.
    q_max = brentq(lambda q: ellipk(q) - field / 4.0,
                   0.0, 1.0 - 1e-14, xtol=1e-14)

    def parameters(q):
        K = ellipk(q)
        b2 = (4.0 * K / field)**2
        a2 = q * b2
        return K, a2, b2

    def counting_field(q):
        K, a2, b2 = parameters(q)
        # Complete elliptic integral of the third kind Pi(a^2 | q).
        Pi = (elliprf(0.0, 1.0 - q, 1.0)
              + a2 * elliprj(0.0, 1.0 - q, 1.0, 1.0 - a2) / 3.0)
        return field * np.sqrt(max(0.0, (1.0-a2)*(1.0-b2))) * Pi / K

    if k == 0.0:
        q = q_max
    else:
        # Explicit endpoint values avoid roundoff at b=1.
        def constraint(q):
            if q == q_max:
                return -k
            return counting_field(q) - k
        q = brentq(constraint, 0.0, q_max, xtol=1e-14)
    K, a2, b2 = parameters(q)
    mean_m2 = b2 * (1.0 - ellipe(q) / K)
    return field**2 * (2.0 * mean_m2 - a2 - b2) / 4.0


def predict_at(experiments, diffusivity):
    """Return raw SCGF values in inverse microscopic time units."""
    return float(diffusivity) * np.asarray([
        _unit_scgf(float(e['field']), float(e['bias']))
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Bounded Gaussian maximum likelihood with known independent errors."""
        records = list(records)
        if not records:
            return self
        basis = predict_at([r['input'] for r in records], 1.0)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError('Calibration values must be finite and sigma positive.')
        weighted_basis = basis / sigma
        information = weighted_basis @ weighted_basis
        if information > 0:
            estimate = (weighted_basis @ (values / sigma)) / information
            self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
