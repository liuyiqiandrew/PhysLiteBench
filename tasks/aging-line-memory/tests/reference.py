from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre
from scipy.special import i0, i1


@lru_cache(None)
def glass(t1, s1, t2, s2, tg, n, period):
    nodes, w = roots_legendre(16)
    edges = np.linspace(-36, 168, 69)
    z = (edges[:-1, None] + edges[1:, None]) / 2 + (edges[1] - edges[0]) * nodes / 2
    v = np.exp(z.ravel())
    measure = np.tile(w, len(edges) - 1) * v * (edges[1] - edges[0]) / 2
    h1 = s1 / (s1 + v ** (tg / t1))
    h2 = s2 / (s2 + v ** (tg / t2))
    om = 2 * np.pi * np.fft.fftfreq(n, d=period / n)
    u = np.expm1(1j * om[:, None] * h1)
    b = np.expm1(1j * om[:, None] * h2)
    exponent = (u @ measure)[:, None] + (b @ measure)[None, :] + (u * measure) @ b.T
    x0 = -period / 4
    cf = np.exp(exponent - 1j * x0 * (om[:, None] + om[None, :]))
    grid = np.arange(n) * period / n + x0
    prob = np.fft.fft2(cf).real / n**2
    # Smooth extension off the physical support of H for Fourier interpolation.
    low = np.minimum(grid + 0.4, 0)
    gg = np.where(grid < -0.4, -0.4 + 0.24 * np.tanh(low / 0.24), grid)
    x = np.clip((grid - x0) / (period / 8), 0.0, 1.0)
    taper = x**4 * (35 - 84 * x + 70 * x**2 - 20 * x**3)
    return gg * taper + (0.75 * period) * (1 - taper), prob


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


def truth(q, tg, n=512, period=32):
    p1, p2 = q["exposures"]
    t1, s1 = p1["temperature"], p1["clock"]
    t2, s2 = p2["temperature"], p2["clock"]
    grid, P = glass(t1, s1, t2, s2, tg, n, period)
    lo, hi = q["pump_band"]
    a, w = roots_legendre(36)
    v = (hi + lo + (hi - lo) * a) / 2
    h = s1 / (s1 + v ** (tg / t1))
    hp = s2 / (s2 + np.exp(-q["shift"] / t2) * v ** (tg / t2))
    F = terminal(p1, grid[:, None] + h)
    G = terminal(p2, grid[:, None] + hp)
    weighted = (P.T @ F["tagged"]) * (h * w * (hi - lo) / 2)
    den = np.sum(weighted * G["total"])
    num = np.sum(weighted * G[q["read"]] * (hp if q["read"] == "tagged" else 1.0))
    return float(num / den), float(den)
