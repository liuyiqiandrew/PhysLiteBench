"""Counting statistics of recoiling branching Cauchy carriers on the unit interval."""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from functools import lru_cache


@lru_cache(None)
def nodes(n):
    g, w = np.polynomial.legendre.leggauss(n)
    return g, w


def mesh(q, n):
    thresholds = q["thresholds"]
    points = sorted(
        set(
            [-1.0, 1.0]
            + [
                z
                for s in (-q["recoil"], q["recoil"])
                for p in [-1.0, 1.0, -1.0 - thresholds[0], 1.0 + thresholds[1]]
                if -1.0 < (z := p - s) < 1.0
            ]
        )
    )
    angles = np.arccos(points[::-1])
    g, w = nodes(n)
    a = angles[:-1]
    b = angles[1:]
    # A cosine map on each panel resolves both endpoint square roots of recoiling daughters.
    u = (1.0 + g) * np.pi / 2
    v = (1.0 - np.cos(u)) / 2
    return (a[:, None] + (b - a)[:, None] * v).ravel(), (
        (b - a)[:, None] * (np.sin(u) * w * np.pi / 4)
    ).ravel()


def channels(left, right):
    """Two mutually exclusive colors on a single final position."""
    return np.stack(
        [left, right, left + right, left, right, np.zeros_like(left)], axis=-1
    )


def leaves(q, z):
    z = np.asarray(z)
    t1, t2 = q["thresholds"]
    a, b = q["efficiencies"]
    return channels(a * (z <= -1 - t1), b * (z >= 1 + t2))


def pair(left, right):
    out = np.empty_like(left)
    out[..., :3] = left[..., :3] + right[..., :3] - left[..., :3] * right[..., :3]
    out[..., 3:] = left[..., 3:] * (1 - right[..., 2, None]) + right[..., 3:] * (
        1 - left[..., 2, None]
    )
    out[..., 5] += left[..., 3] * right[..., 4] + right[..., 3] * left[..., 4]
    return out


def emit(h):
    total = h[0] + h[1] - h[2]
    return (
        np.array([h[0], total / h[0], h[5] / total])
        if total > 1e-12
        else np.array([h[0], 0.0, np.nan])
    )


def solve(q, D, n=52, shortcut=None):
    rate = q["rate"]
    dist = q["recoil"]
    th, w = mesh(q, max(42, n + 4))
    x = np.cos(th)
    m = np.arange(1, n + 1)
    basis = np.sin(th[:, None] * m)
    proj = (2 / np.pi) * (basis * (np.sin(th) * w)[:, None]).T

    def ix(z):
        z = np.asarray(z)
        inside = abs(z) < 1
        return np.sin(np.arccos(np.clip(z, -1.0, 1.0))[..., None] * m) * inside[
            ..., None
        ], leaves(q, z)

    G = lu_factor(np.diag(D * m) + rate * (proj @ basis))
    P = lu_solve(G, proj)
    t1, t2 = q["thresholds"]
    wa, wb = q["efficiencies"]
    source = channels(wa * D / (np.pi * (x + 1 + t1)), wb * D / (np.pi * (1 + t2 - x)))
    base = P @ source
    X1, E1 = ix(x - dist)
    X2, E2 = ix(x + dist)

    def branch(left, right):
        if shortcut == "equal recoil":
            return (pair(left, left) + pair(right, right)) / 2.0
        return pair(left, right)

    C = np.zeros((n, 6))
    for step in range(180):
        new = base + rate * P @ branch(X1 @ C + E1, X2 @ C + E2)
        if np.max(abs(new - C)) < 8e-13:
            break
        C = new
    else:
        raise RuntimeError("uncertified fixed point")

    def local(z):
        X, E = ix(z)
        return X @ new + E

    start = q["launch"]
    h = (
        local(start)
        if q["source"] == "single"
        else pair(local(start - dist), local(start + dist))
    )
    return emit(h)
