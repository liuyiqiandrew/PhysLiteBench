"""Free phase-space flux and a bound on the omitted low-speed tail."""
import numpy as np
from scipy.special import erf, gamma, roots_legendre
from renewal import density, geometry, slow_probability


def boundary_flux(t, u, v, D):
    return (
        np.sqrt(3)
        * v
        / (2 * np.pi * D * t * t)
        * np.exp(-(u * u - u * v + v * v) / (D * t))
    )


def renewal_check(t, u, v, D, order):
    z, w = roots_legendre(order)
    y = (z + 1) / 2
    s = t * y[:, None]
    x = (y / (1 - y))[None, :]
    hit = boundary_flux(s, u, x, D) * erf(np.sqrt(3 * u * x / (D * s)))
    second = boundary_flux(t - s, -x, v, D)
    correction = (
        t / 4 * np.sum(w[:, None] * w[None, :] * hit * second / (1 - y)[None, :] ** 2)
    )
    full = boundary_flux(t, u, v, D)
    first = full * erf(np.sqrt(3 * u * v / (D * t)))
    return full, first, correction


def tail_bound(q, D):
    # 1-exp(-x) <= x**a. Subadditivity bounds the rest of the heat/time
    # perpetuity by a geometric series in E[(rR)**(2a)].
    z = np.arange(-30, 155, 0.04)
    r, _, k, c = geometry(z, q["threshold"], D)
    p = slow_probability(c, k)
    f, s = q["restitution"]
    power = 0.25
    m = 0.04 * np.sum(density(z) * r**power * ((1 - p) * f**power + p * s**power))
    moment = 1 / (2 * np.cos(np.pi * (1 + power) / 3))
    a = power / 2
    step_moment = (q["clock"] / D) ** a * gamma(1 - a) * (1 + moment) + (
        q["dose"] / 2
    ) ** a * moment
    return (q["launch"] * np.exp(-65)) ** power * step_moment / (1 - m)
