"""Fourier inversion of the compound Poisson spectra in the two reporter fields."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre


X = np.linspace(-24.0, 24.0, 3072, endpoint=False)
K = 2 * np.pi * np.fft.fftfreq(len(X), X[1] - X[0])
NODES, WEIGHTS = roots_legendre(220)


def spectrum(transform, atom):
    """Invert E[exp(ikF)] and E[exp(ikF-iS)] for one pocket, keeping its zero-shift atom."""
    density = np.fft.fftshift(np.fft.fft(transform - atom[:, None], axis=1), axes=1).T / 48.0
    return atom, CubicSpline(X, density, extrapolate=False)


def single_defect(kicks, direction, sigma, clock, forced=()):
    """Characteristic functions of one Gaussian defect given a free clock and imposed flips.

    clock(t) is the integrated intensity of undetected flips up to t. Forced times
    reverse the same polarity in addition to the free flips.
    """
    paths = [(1, 0.0, 1.0)]
    last = 0.0
    for t, a, b in kicks:
        same = 0.5 * (1 + np.exp(-2 * (clock(t) - clock(last))))
        sign = (-1) ** sum(f < t for f in forced)
        area = sign * (a * direction[0] + b * direction[1])
        paths = [
            (s * d, h + s * d * area, p * w)
            for s, h, p in paths
            for d, w in [(1, same), (-1, 1 - same)]
        ]
        last = t
    marked = sum(p * np.exp(-0.5 * sigma**2 * (K - h) ** 2) for _, h, p in paths)
    return np.array([np.exp(-0.5 * (sigma * K) ** 2), marked])


def collect(q, sources):
    """Condition a mixture of independent-pocket measures on the two optical slit centers.

    Each entry (w, r, pockets) carries raw weight w, overlap ratio r, and three
    pocket (atom, density) pairs from spectrum(). The two columns are preparation
    counts and Ramsey coherence. Mixture weights need not sum to one.
    """
    a, b = q["gate"]
    u, v = q["mix"]
    det = 1 - u * v
    candidates = {(a, b), (a, -b)} if q["unsigned_b"] else {(a, b)}
    total = np.zeros((3, 2), complex)
    for w, r, fields in sources:
        (ax, fx), (ay, fy), (az, fz) = fields
        for m1, m2 in candidates:
            x, y = (m1 - u * m2) / det, (m2 - v * m1) / det
            if abs(x) < 1e-10 and abs(y) < 1e-10:
                total[0] += w * ax * ay * az
            for j, (coord, residual, scale) in enumerate(
                [
                    (x, y, max(1.0, abs(v))),
                    (y, x, max(1.0, abs(u))),
                    (x, y - r * x, max(abs(1 + u * r), abs(v + r))),
                ]
            ):
                if abs(residual) < 1e-10:
                    total[1] += (
                        w
                        * fields[j][1](coord)
                        * fields[(j + 1) % 3][0]
                        * fields[(j + 2) % 3][0]
                        / scale
                    )
            two = (
                az * fx(x) * fy(y)
                + ax * fz(x) * fy(y - r * x)
                + ay * fz(y / r) * fx(x - y / r) / abs(r)
            )
            z = 11.0 * NODES
            two += 11.0 * (WEIGHTS @ (fz(z) * fx(x - z) * fy(y - r * z)))
            total[2] += w * two / abs(det)
    for den, num in total:
        if den.real > 1e-13:
            return float((np.exp(-1j * q["phase"]) * num).real / den.real)
    raise ValueError("zero accepted intensity")


def passive(q):
    if q.get("blank"):
        return np.cos(q["phase"])
    sources = []
    for domain in q["domains"]:
        fields = []
        r = domain["ratio"]
        for (n, s, g), d in zip(domain["pockets"], [(1, 0), (0, 1), (1, r)]):
            one = single_defect(q["kicks"], d, s, lambda t: g * t)
            fields.append(spectrum(np.exp(n * (one - 1)), np.full(2, np.exp(-n))))
        sources.append((domain["weight"], r, fields))
    return collect(q, sources)
