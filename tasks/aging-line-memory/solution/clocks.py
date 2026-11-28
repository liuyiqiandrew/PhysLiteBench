from decimal import Decimal, localcontext
from math import comb, gamma
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import i0, i1, roots_legendre


@lru_cache(None)
def background(alpha, rate, order=20):
    """Gaussian quadrature of the quenched sum of individual readout probabilities."""
    with localcontext() as cx:
        cx.prec = 110
        a = Decimal(str(alpha))
        first = Decimal(str(rate**alpha * gamma(1 + alpha) * gamma(1 - alpha)))
        cumulants = [Decimal(0), first]
        moments = [Decimal(1)]
        for n in range(1, 2 * order + 1):
            if n > 1:
                cumulants.append(cumulants[-1] * (n - 1 - a) / (n - 1))
            moments.append(
                sum(
                    comb(n - 1, j - 1) * cumulants[j] * moments[n - j]
                    for j in range(1, n + 1)
                )
            )

        def dot(p, q, shift=0):
            return sum(
                x * y * moments[i + j + shift]
                for i, x in enumerate(p)
                for j, y in enumerate(q)
            )

        old = []
        poly = [Decimal(1)]
        norm_prev = Decimal(1)
        diagonal = []
        off = []
        for j in range(order):
            norm = dot(poly, poly)
            delta = dot(poly, poly, 1) / norm
            ratio = norm / norm_prev
            diagonal.append(float(delta))
            if j:
                off.append(float(ratio.sqrt()))
            next_poly = [Decimal(0), *poly]
            for i, x in enumerate(poly):
                next_poly[i] -= delta * x
            for i, x in enumerate(old):
                next_poly[i] -= ratio * x
            old, poly, norm_prev = poly, next_poly, norm
    x, v = eigh_tridiagonal(diagonal, off)
    return x, v[0] ** 2


def readings(q, tg, reset=False):
    a = q["temperature"] / tg
    s = q["clock"]
    r = np.exp(-np.array(q["offsets"]) / q["temperature"])
    x, y = q["levels"]
    hs, wt = background(a, s)
    band = q.get("band")
    if band:
        z, weights = roots_legendre(32)
        lo, hi = band
        h = s / (s + (0.5 * (hi + lo + (hi - lo) * z)) ** (1 / a))
        measure = 0.5 * (hi - lo) * weights * h

    def op(m):
        m = np.asarray(m)
        u, v = r[0] * m, r[1] * m
        base = np.exp(-(s + r[0] - u) * x - (s + r[1] - v) * y)
        if y == 0:
            return base, 0 * base
        z = np.sqrt(u * v * x * y)
        return base * u * v * x * i1(2 * z) / z, base * u * i0(2 * z)

    m = 1 / (2 + hs)
    mean = wt @ m
    wA, wB = op(mean if reset else m)
    weights = np.ones(1) if reset else wt
    m0 = mean if reset else m
    den = np.sum(weights * (s * (wA + wB) + (1 - 2 * m0) * (r[0] * wA + r[1] * wB)))
    if q["read"] in ("A", "B"):
        num = s * np.sum(weights * (wA if q["read"] == "A" else wB))
    elif not band:
        num = den - s * np.sum(weights * (wA + wB))
    else:
        shifted = 1 / (2 + hs[:, None] + h)
        if not reset:
            wA, wB = op(shifted)
        num = wt @ (shifted * (r[0] * wA + r[1] * wB)) @ measure
    return float(num / den), float(den)
