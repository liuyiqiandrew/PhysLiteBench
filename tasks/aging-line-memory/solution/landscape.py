"""Three or four line samplings from a single frozen glass."""
from functools import lru_cache
import math
import numpy as np
from scipy.special import roots_legendre, roots_jacobi


@lru_cache(None)
def blocks(n):
    if n == 0:
        return ((),)
    out = []
    for old in blocks(n - 1):
        out.append(old + ((n - 1,),))
        for i, h in enumerate(old):
            out.append(old[:i] + (h + (n - 1,),) + old[i + 1 :])
    return tuple(out)


def density(v, a):
    return v ** (-1 / a)


def match(parts, i, j):
    return any(i in g and j in g for g in parts)


def marked(q, tg, nz=150, nmix=58):
    (T, H), (T2, H2) = q["settings"]
    a, b = T / tg, T2 / tg
    draws = q["preparations"]
    n = len(draws)
    na = draws.count(0)
    nb = n - na
    x, wt = roots_jacobi(nmix, na / a - 1, nb / b - 1)
    p = (x + 1) / 2
    wt = wt * 2 ** (1 - na / a - nb / b) / (a * b * math.gamma(na) * math.gamma(nb))
    k, raw = roots_legendre(nz)
    t, tail = roots_legendre(70)
    z = np.exp(np.r_[10 * k + 1, 39 * t + 50])
    zw = z * np.r_[10 * raw, 39 * tail]
    # Keep a common scaled depth for both sample-specific partition functions.
    moment = np.array([-1, 1])
    ua = density(z, a)[None, :] * np.exp(H * moment[:, None] / T)
    ub = density(z, b)[None, :] * np.exp(H2 * moment[:, None] / T2)
    h = (1 - p[:, None, None]) ** (1 / a) * ua[None, :, :] + p[:, None, None] ** (
        1 / b
    ) * ub[None, :, :]
    mass = np.exp(-9) + (-np.expm1(-h)).mean(1) @ zw
    live = np.exp(-h) * zw
    dur = q["exposure"]
    seen = {}

    def m(group, l):
        key = (group, l)
        if key not in seen:
            a1 = sum(draws[i] == 0 for i in group)
            b1 = len(group) - a1
            terms = live * (ua**a1 * ub**b1)[None, :, :]
            if q.get("tag") and 0 in group:
                sign = q["tag"]
                terms = terms * (moment == sign)[None, :, None]
            if n - 1 in group:
                terms = terms / (mass[:, None, None] + dur * z[None, None, :]) ** l
            seen[key] = terms.sum(2).mean(1)
        return seen[key]

    num = denom = 0.0
    for pattern in blocks(n):
        if any(match(pattern, i, j) != same for i, j, same in q["gate"]):
            continue
        l = len(pattern)
        vec = math.gamma(l) * wt
        for g in pattern:
            vec = vec * m(g, l)
        prob = vec.sum()
        denom += prob
        if match(pattern, q["mark"], n - 1):
            num += prob
    return num / denom, denom
