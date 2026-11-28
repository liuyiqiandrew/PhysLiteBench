"""Positive multiplicities and direct binomial labellings of the physical conductors."""
import math
import numpy as np
from functools import lru_cache
from scipy.special import comb


@lru_cache(None)
def classes(r, t):
    states = []
    # Coefficient of z^n in cos(2k arccos(sqrt(z)/2)), using the exact
    # polynomial coefficients of T_{2k}, then remove the constant term.
    for b in range(-8, 9):
        for a in range(9):
            if math.gcd(a, abs(b)) != 1 or a == 0 and b < 1:
                continue
            ell = (a + b * t) ** 2 / r + b * b * r
            for n in range(1, 15):
                terms = []
                for k in range(n, 19):
                    coef = k * (-1) ** (k - n) * math.comb(k + n, k - n) / (k + n)
                    terms.append(2 * coef * np.exp(-2 * np.pi * ell * k * k / 3))
                w = math.fsum(terms)
                assert w >= -1e-16
                if w > 1e-16:
                    states.append((a, b, n, w))
    return states


def value(q, gain):
    den = num = 0.0
    gx, gy, m, l = q["gate"]
    for a, b, n, w in classes(q["aspect"], q["shear"]):
        p = 1 / (1 + np.exp(-2 * gain * np.dot(q["bias"], [a, b])))
        for j in range(n + 1):
            signed = 2 * j - n
            if signed * (gx * a + gy * b) % m != l:
                continue
            prob = w * math.comb(n, j) * p**j * (1 - p) ** (n - j)
            den += prob
            num += prob * np.cos(q["phase"] + signed * np.dot(q["probe"], [a, b]))
    return num / den, den / sum(j[3] for j in classes(q["aspect"], q["shear"]))
