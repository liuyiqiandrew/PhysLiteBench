"""Real-frequency inversion using a series over visits between the two walls."""
from decimal import Decimal, localcontext
from functools import lru_cache
from math import factorial
import numpy as np
from scipy.special import roots_legendre, ndtr


@lru_cache
def stehfest(order):
    coeff = []
    for k in range(1, 2 * order + 1):
        weight = Decimal(0)
        for j in range((k + 1) // 2, min(k, order) + 1):
            num = j**order * factorial(2 * j)
            den = (
                factorial(order - j)
                * factorial(j)
                * factorial(j - 1)
                * factorial(k - j)
                * factorial(2 * j - k)
            )
            weight += Decimal(num) / Decimal(den)
        coeff.append((-1) ** (k + order) * weight)
    return coeff


def image(q, gamma, terms=18):
    L = q["circumference"]
    D = 2 * gamma
    T = q["height"]
    t = T * q["level"]
    a, b = q["clocks"]
    b0, b1 = q["bottom"]
    top = q["top"]
    winding = q["winding"]
    f0 = top[winding % 2] + (winding // 2) * L
    f1 = top[(winding + 1) % 2] + ((winding + 1) // 2) * L
    start = b1 - b0
    end = f1 - f0
    mid = ((b0 + b1) * (1 - t / T) + (f0 + f1) * t / T) / 2
    sig = np.sqrt(gamma * t * (1 - t / T))
    xx, ww = roots_legendre(56)
    x = (1 + xx) * L / 2
    ww *= L / 2

    def free(s, dt):
        return np.exp(-s * s / (4 * D * dt)) / np.sqrt(4 * np.pi * D * dt)

    def images(x0, x1, dt, sign):
        shift = 2 * L * np.arange(-9, 10)
        return np.sum(
            free(np.asarray(x1 - x0)[..., None] + shift, dt)
            + sign * free(np.asarray(x1 + x0)[..., None] + shift, dt),
            axis=-1,
        )

    lo, hi = q["window"]
    shifts = np.arange(-7, 8) * L
    loc = mid + (-0.5 if q["tag"] == 0 else 0.5) * x
    tag = np.sum(
        ndtr((hi + shifts - loc[:, None]) / sig)
        - ndtr((lo + shifts - loc[:, None]) / sig),
        axis=1,
    )
    sampled = ww * tag * images(x, end, T - t, 1) / images(start, end, T, 1)
    if a == 0 and b == 0:
        return float(sampled @ images(start, x, t, -1))

    def dec(v):
        return Decimal(str(v))

    with localcontext() as ctx:
        ctx.prec = 2 * terms + 22
        width, diffusion, initial = map(dec, [L, D, start])
        A, B = dec(a) * width, dec(b) * width
        freq = Decimal(2).ln() / dec(t)
        nodes = list(map(dec, x))
        measure = list(map(dec, sampled))

        def density(p):
            z = (p / diffusion).sqrt()
            pos = (z * width).exp()
            v = 2 / (pos - 1 / pos)
            h = z * v
            c = z * v * (pos + 1 / pos) / 2

            def edges(y):
                p0 = (z * y).exp()
                p1 = pos / p0
                return v * (p1 - 1 / p1) / 2, v * (p0 - 1 / p0) / 2

            l0, r0 = edges(initial)
            l1 = r1 = Decimal(0)
            for y, w in zip(nodes, measure):
                front, back = edges(y)
                l1 += front * w
                r1 += back * w
            if a == 0 or b == 0:
                return (
                    (l0 * l1 if a > 0 else r0 * r1)
                    * width
                    / diffusion
                    * (-c * (A + B)).exp()
                )
            xab = h * h * A * B
            series = off = diag = Decimal(1)
            for j in range(1, 100):
                series *= xab / (j * j)
                off += series
                diag += series / (j + 1)
                if series < Decimal("1e-52") * off:
                    break
            return (
                width**2
                / diffusion
                * (-c * (A + B)).exp()
                * h
                * ((l0 * r1 + r0 * l1) * off + h * (A * l0 * l1 + B * r0 * r1) * diag)
            )

        return float(
            freq * sum(w * density(j * freq) for j, w in enumerate(stehfest(terms), 1))
        )
