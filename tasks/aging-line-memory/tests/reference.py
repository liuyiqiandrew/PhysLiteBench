"""Distribution of the ordinary-site return sum and explicit alternating paths."""
from functools import lru_cache
from math import gamma
import numpy as np
from scipy.special import roots_jacobi, roots_legendre


@lru_cache(None)
def circle(alpha, rate, size, period):
    # The removable 0/0 in older SciPy Jacobi recurrences has an analytic limit.
    with np.errstate(invalid="ignore"):
        e, w = roots_jacobi(256, alpha - 1, -alpha)
    assert np.isfinite(e).all() and np.isfinite(w).all()
    h = (e + 1) / 2
    w /= w.sum()
    omega = np.arange(size // 2 + 1) * 2 * np.pi / period
    ex = np.expm1(1j * omega[:, None] * h) @ (w / h)
    cf = np.exp(rate**alpha * gamma(1 + alpha) * gamma(1 - alpha) * ex)
    mass = np.fft.irfft(cf.conj(), n=size)
    return np.arange(size) * period / size, mass


def truth(q, tg, size=4096, period=48):
    alpha = q["temperature"] / tg
    s = q["clock"]
    H, prob = circle(alpha, s, size, period)
    rA, rB = np.exp(-np.array(q["offsets"]) / q["temperature"])
    x, y = q["levels"]

    def ends(H):
        m = 1 / (2 + H)
        u, v = rA * m, rB * m
        base = np.exp(-(s + rA - u) * x - (s + rB - v) * y)
        if y == 0:
            return base, base * 0.0
        wA = u * v * x
        wB = u.copy()
        a, b = wA.copy(), wB.copy()
        for n in range(1, 26):
            wA = wA * (u * v * x * y / (n * (n + 1)))
            wB = wB * (u * v * x * y / n**2)
            a += wA
            b += wB
        return base * a, base * b

    A, B = ends(H)
    total = prob @ (s * (A + B) + (rA * A + rB * B) * H / (2 + H))
    if q["read"] == "A":
        n = s * (prob @ A)
    elif q["read"] == "B":
        n = s * (prob @ B)
    elif "band" not in q:
        n = total - s * prob @ (A + B)
    else:
        lo, hi = q["band"]
        z, w = roots_legendre(35)
        marks = (lo + hi + (hi - lo) * z) / 2
        hv = s / (s + marks ** (1 / alpha))
        H = H[:, None] + hv[None, :]
        A, B = ends(H)
        n = (prob @ ((rA * A + rB * B) / (2 + H))) @ (w * hv) * (hi - lo) / 2
    return float(n / total), float(total)
