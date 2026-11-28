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


def terminal(p, H):
    s = p["clock"]
    rA, rB = np.exp(-np.array(p["offsets"]) / p["temperature"])
    x, y = p["levels"]
    m = 1 / (2 + H)
    u = rA * m
    v = rB * m
    pref = np.exp(-(s + rA - u) * x - (s + rB - v) * y)
    if y == 0:
        A = pref
        B = 0.0 * A
    else:
        z = np.sqrt(u * v * x * y)
        A = pref * u * v * x * i1(2 * z) / z
        B = pref * u * i0(2 * z)
    rate = m * (rA * A + rB * B)
    return dict(
        A=s * A, B=s * B, host=H * rate, total=s * (A + B) + H * rate, tagged=rate
    )


def independent(q, tg):
    p1, p2 = q["exposures"]
    H, w = background(p1["temperature"] / tg, p1["clock"])
    K, z = background(p2["temperature"] / tg, p2["clock"])
    u, weights = roots_legendre(32)
    lo, hi = q["pump_band"]
    v = (hi + lo + (hi - lo) * u) / 2
    h = p1["clock"] / (p1["clock"] + v ** (tg / p1["temperature"]))
    hp = p2["clock"] / (
        p2["clock"]
        + np.exp(-q["shift"] / p2["temperature"]) * v ** (tg / p2["temperature"])
    )
    pumped = w @ terminal(p1, H[:, None] + h)["tagged"]
    after = terminal(p2, K[:, None] + hp)
    gates = weights * h * pumped
    score = (z @ after[q["read"]]) * (hp if q["read"] == "tagged" else 1.0)
    return float((gates @ score) / (gates @ (z @ after["total"])))
