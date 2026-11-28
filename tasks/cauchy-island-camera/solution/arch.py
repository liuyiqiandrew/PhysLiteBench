"""Boundary trace generator and time-resolved strip-to-floor excursion currents."""
from functools import lru_cache
import numpy as np
from scipy.special import erfc, erfcx
from scipy.linalg import solve


@lru_cache(None)
def modes(n):
    g, w = np.polynomial.legendre.leggauss(max(100, 2 * n + 10))
    t = (1 + g) * np.pi / 2
    w = w * np.pi / 2 * np.sin(t)
    m = np.arange(1, n + 1)
    return np.cos(t), np.sin(t[:, None] * m), w, m


def kernel(d, h):
    if h is None:
        return 1 / (np.pi * d * d)
    if h == 0:
        return np.zeros_like(d)
    a = np.pi * abs(d) / (2 * h)
    return np.pi / (4 * h * h) / np.sinh(a) ** 2


def defect(d, h):
    u = np.pi * d / (2 * h)
    out = np.zeros_like(d)
    mask = abs(u) > 0.025
    out[mask] = 1 / (np.pi * d[mask] ** 2) - kernel(d[mask], h)
    v = u[~mask]
    out[~mask] = np.pi / (4 * h * h) * (1 / 3 - v * v / 15 + 2 * v**4 / 189)
    return out


def timeflux(d, h, t):
    if h == 0:
        return np.zeros_like(d)
    gamma = (np.pi / h) * np.arange(1, int(np.ceil(34 * h / (np.pi * np.min(d)))) + 1)[
        :, None
    ]
    p = d[None, :] / (2 * np.sqrt(t))
    u = gamma * np.sqrt(t)
    v = gamma * d
    half = (np.exp(-v) * erfc(p - u) - np.exp(-p * p - u * u) * erfcx(p + u)) / 2
    return (gamma * half).sum(axis=0) / h


def reading(q, D, n=50, uncoupled=False):
    z, basis, w, m = modes(n)
    start = q["launch"]
    y = q["landing"]
    d = abs(y - z)
    lo, hi = q["heights"]
    prior = q["prior"]
    last = kernel(d, hi) - kernel(d, lo)
    plain = kernel(d, None)
    dt0, dt1 = np.array(q["window"]) * D
    if uncoupled:
        timed = last * (np.exp(-d * d / (4 * dt1)) - np.exp(-d * d / (4 * dt0)))
    else:
        timed = (timeflux(d, hi, dt1) - timeflux(d, lo, dt1)) - (
            timeflux(d, hi, dt0) - timeflux(d, lo, dt0)
        )
    rhs = (2 / np.pi) * (basis * w[:, None]).T @ np.stack([plain, last, timed], axis=1)
    init = np.sin(m * np.arccos(start))
    unbounded = (init / m) @ rhs
    if prior is None:
        filtered = unbounded
    else:
        Q = basis * w[:, None]
        mat = np.diag(m) + (2 / np.pi) * Q.T @ defect(z[:, None] - z, prior) @ Q
        filtered = init @ solve(mat, rhs, check_finite=False)
    return np.array(
        [
            unbounded[1] / unbounded[0],
            filtered[1] / unbounded[1],
            filtered[2] / filtered[1],
        ]
    )
