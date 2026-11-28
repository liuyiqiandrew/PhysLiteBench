"""Boundary-clock propagator of a reflecting gap on an interval."""
import numpy as np
from scipy.special import roots_legendre, iv, ndtr


def gap_kernel(x, y, t, L, D, killed=False):
    k = np.arange(1, 56) * np.pi / L
    trig = np.sin if killed else np.cos
    return (
        (0 if killed else 1)
        + 2
        * np.sum(
            trig(np.asarray(x)[..., None] * k)
            * trig(np.asarray(y)[..., None] * k)
            * np.exp(-D * t * k * k),
            axis=-1,
        )
    ) / L


def camera(q, gamma):
    L = q["circumference"]
    x0, x1 = q["bottom"]
    t = q["level"] * q["height"]
    T = q["height"]
    end = [
        q["top"][(i + q["winding"]) % 2] + L * ((i + q["winding"]) // 2)
        for i in range(2)
    ]
    L0, L1 = q["clocks"]
    D = 2 * gamma
    g0 = x1 - x0
    gT = end[1] - end[0]
    frac = q["level"]
    mid = ((1 - frac) * (x0 + x1) + frac * sum(end)) / 2
    sd = np.sqrt(gamma * T * frac * (1 - frac))
    y, w = roots_legendre(44)
    y = (1 + y) * L / 2
    w = w * L / 2
    shift = np.arange(-6, 7) * L
    loc = mid + y[:, None] * (q["tag"] - 0.5)
    lo, hi = q["window"]
    p = np.sum(ndtr((hi + shift - loc) / sd) - ndtr((lo + shift - loc) / sd), axis=1)
    measure = w * p * gap_kernel(y, gT, T - t, L, D) / gap_kernel(g0, gT, T, L, D)
    if L0 == 0 and L1 == 0:
        return float(measure @ gap_kernel(g0, y, t, L, D, True))
    # Talbot's time contour. Lengths of positive camera readings are scaled by L.
    M = 22
    theta = np.pi * np.arange(1, M) / M
    cot = 1 / np.tan(theta)
    r = 0.4 * M
    z = np.r_[r, r * theta * (cot + 1j)]
    weights = np.exp(z) * np.r_[0.5, 1 + 1j * (theta * (1 + cot * cot) - cot)]
    alpha = np.sqrt(z / (t * D))
    c = alpha / np.tanh(alpha * L)
    d = alpha / np.sinh(alpha * L)

    def boundary(x):
        return np.sinh(alpha[:, None] * x) / np.sinh(alpha * L)[:, None]

    left = boundary(L - y) @ measure
    right = boundary(y) @ measure
    s = np.sinh(alpha * (L - g0)) / np.sinh(alpha * L)
    h = np.sinh(alpha * g0) / np.sinh(alpha * L)
    if L0 == 0 or L1 == 0:
        v = (s * left if L0 else h * right) * np.exp(-c * L * (L0 + L1)) * L / D
    else:
        A, B = L * L0, L * L1
        arg = 2 * d * np.sqrt(A) * np.sqrt(B)
        scaled = np.ones_like(arg)
        np.divide(2 * iv(1, arg), arg, out=scaled, where=abs(arg) > 1e-8)
        diag = d * scaled * (A * s * left + B * h * right)
        off = iv(0, arg) * (s * right + h * left)
        v = d * np.exp(-c * L * (L0 + L1)) * (diag + off) * L**2 / D
    return float(0.4 / t * np.real(weights @ v))
