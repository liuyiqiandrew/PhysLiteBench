from functools import lru_cache
from math import factorial
import numpy as np
from scipy.special import roots_laguerre, roots_legendre


def gauss(n, lo, hi):
    z, w = roots_legendre(n)
    return (hi + lo + (hi - lo) * z) / 2, (hi - lo) * w / 2


@lru_cache(None)
def quenched(t1, s1, t2, s2, tg, res, correlated):
    z, w = roots_laguerre(res)
    u = np.r_[0.0, z / 2]
    wu = np.r_[1.0, w / 2]
    # Unit-intensity Poisson process in depth rank v. Integrate on log v.
    a1, a2 = np.array([t1, t2]) / tg
    logv = np.arange(-34.0, 160.0, 0.093)
    v = np.exp(logv)
    measure = 0.093 * v
    h1 = s1 / (s1 + np.exp(logv / a1))
    h2 = s2 / (s2 + np.exp(logv / a2))
    A = -np.expm1(-u[:, None] * h1)
    B = -np.expm1(-u[:, None] * h2)
    psi = (A @ measure)[:, None] + (B @ measure)[None, :]
    if correlated:
        psi -= (A * measure) @ B.T
    return u, wu, np.exp(-psi)


def transforms(p, u, order=30):
    s = p["clock"]
    rA, rB = np.exp(-np.asarray(p["offsets"]) / p["temperature"])
    x, y = p["levels"]
    c = rA * x + rB * y
    # Exponential times the alternating residence histories, in powers of m.
    n = np.arange(order + 1)
    expo = np.array([c ** int(k) / factorial(int(k)) for k in n])
    aa = np.zeros(order + 1)
    bb = aa.copy()
    if y == 0:
        aa[0] = 1.0
    else:
        ab = rA * rB * x * y
        for k in range(order // 2):
            bb[2 * k + 1] = rA * ab**k / (factorial(k) ** 2)
            aa[2 * k + 2] = rA * rB * x * ab**k / (factorial(k) * factorial(k + 1))
    aa = np.convolve(aa, expo)[: order + 1]
    bb = np.convolve(bb, expo)[: order + 1]
    v = rA * aa + rB * bb
    excursion = np.r_[0.0, v[:-1]]
    host = v - 2 * excursion
    total = host + s * (aa + bb)
    powers = np.array([u[1:] ** int(k) / factorial(int(k)) for k in n[:-1]], float)

    def encode(c):
        return np.r_[c[0], c[1:] @ powers]

    return {
        k: encode(c)
        for k, c in [
            ("A", s * aa),
            ("B", s * bb),
            ("host", host),
            ("tagged", excursion),
            ("total", total),
        ]
    }


def pump(q, tg, res=42, control=""):
    p1, p2 = q["exposures"]
    t1, s1 = p1["temperature"], p1["clock"]
    t2, s2 = p2["temperature"], p2["clock"]
    u, w, L = quenched(t1, s1, t2, s2, tg, res, control != "fresh")
    v, weights = gauss(44, *q["pump_band"])
    h1 = s1 / (s1 + v ** (tg / t1))
    delta = 0 if control == "passive" else q["shift"]
    h2 = s2 / (s2 + np.exp(-delta / t2) * v ** (tg / t2))
    f1 = transforms(p1, u)
    f2 = transforms(p2, u)
    left = (f1["tagged"] * w)[:, None] * np.exp(-u[:, None] * h1)
    right = w[:, None] * np.exp(-u[:, None] * h2)
    combined = ((L.T @ left) * right) @ (weights * h1)
    den = combined @ f2["total"]
    if q["read"] == "tagged":
        num = ((L.T @ left) * right) @ (weights * h1 * h2) @ f2["tagged"]
    else:
        num = combined @ f2[q["read"]]
    mass = np.exp(
        -sum(
            (p["clock"] + np.exp(-np.array(p["offsets"]) / p["temperature"]))
            @ np.array(p["levels"])
            for p in [p1, p2]
        )
    )
    return float(num / den), float(den * mass)
