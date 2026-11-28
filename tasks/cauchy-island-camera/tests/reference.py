"""Conformal strip Green function and direct integration of the physical-time heat current."""
import numpy as np
from scipy.special import roots_legendre


def quadrature(a, b, n):
    x, w = roots_legendre(n)
    return (a + b) / 2 + (b - a) / 2 * x, w * (b - a) / 2


def green(x, z, h):
    if h is None:
        theta = np.arccos(x)
        phi = np.arccos(z)
    else:

        def angle(v):
            return np.arccos(
                np.clip(
                    2 * np.expm1((v + 1) * np.pi / h) / np.expm1(2 * np.pi / h) - 1,
                    -1.0,
                    1.0,
                )
            )

        theta = angle(x)
        phi = angle(z)
    return np.log(abs(np.sin((theta + phi) / 2) / np.sin((theta - phi) / 2))) / np.pi


def flux(d, h, duration):
    a = d[:, None]
    t = duration[None, :]
    if h is None:
        return np.exp(-a * a / (4 * t)) / (4 * np.pi * t * t)
    if h == 0:
        return np.zeros((len(d), len(duration)))
    short = duration < 0.22 * h * h
    f = np.empty(len(duration))
    u = duration[short]
    k = np.arange(1, 8)[:, None]
    f[short] = (
        1 + 2 * np.sum((1 - 2 * k * k * h * h / u) * np.exp(-k * k * h * h / u), axis=0)
    ) / (np.sqrt(4 * np.pi * u) * u)
    u = duration[~short]
    r = (np.pi / h) * np.arange(1, 13)[:, None]
    f[~short] = (2 / h) * np.sum(r * r * np.exp(-r * r * u), axis=0)
    return np.exp(-a * a / (4 * t)) / np.sqrt(4 * np.pi * t) * f


def reading(q, D, n=90):
    x = q["launch"]
    y = q["landing"]
    p = q["prior"]
    a, b = q["heights"]
    t0, t1 = np.array(q["window"]) * D
    # Quadratic maps on either side of the launch and both interval tips integrate the logarithmic Green peak.
    th = np.arccos(x)
    s, w = quadrature(0, 1, n)
    sin = np.sin(np.pi * s)
    cos = (1.0 - np.cos(np.pi * s)) / 2
    angles = np.r_[th * cos, th + (np.pi - th) * cos]
    ww = np.r_[th * w * sin * np.pi / 2, (np.pi - th) * w * sin * np.pi / 2] * np.sin(
        angles
    )
    z = np.cos(angles)
    d = abs(y - z)
    gu = green(x, z, None) * ww
    gp = green(x, z, p) * ww

    def evaluate(ta, tb):
        # A log-time map keeps small-time concentrations and an exponentially decaying strip tail resolved.
        t, w = quadrature(np.log(ta), np.log(tb), n)
        t = np.exp(t)
        return (flux(d, b, t) - flux(d, a, t)) @ (w * t)

    half = evaluate(0.00005, 64.0)
    timed = evaluate(t0, t1)
    density = np.sqrt(1 - x * x) / (np.pi * np.sqrt(y * y - 1) * abs(y - x))
    return np.array(
        [gu @ half / density, gp @ half / (gu @ half), gp @ timed / (gp @ half)]
    )
