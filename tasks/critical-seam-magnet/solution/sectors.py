"""Equal time, critical quantum-spin correlation on a ring."""
from itertools import combinations
import numpy as np


def theta(z, r):
    z = np.asarray(z)[..., None]
    m = np.arange(0, 12)
    Q = np.pi * r
    half = np.exp(-Q * (m + 0.5) ** 2)
    full = np.exp(-Q * (m + 1) ** 2)
    odd = np.pi * (2 * m + 1) * z
    even = 2 * np.pi * (m + 1) * z
    return np.stack(
        [
            2 * np.sum((-1.0) ** m * half * np.sin(odd), axis=-1),
            2 * np.sum(half * np.cos(odd), axis=-1),
            1 + 2 * np.sum(full * np.cos(even), axis=-1),
            1 + 2 * np.sum((-1.0) ** (m + 1) * full * np.cos(even), axis=-1),
        ],
        axis=-1,
    )


def normalized_blocks(q):
    x = np.asarray(q["sites"])
    r = q["aspect"]
    n = len(x)
    signs = []
    for plus in combinations(range(1, n), n // 2 - 1):
        v = -np.ones(n)
        v[[0, *plus]] = 1
        signs.append(v)
    v = np.array(signs)
    i, j = np.triu_indices(n, 1)
    E = abs(theta(x[i] - x[j], r)[:, 0])
    factors = np.exp(0.5 * (v[:, i] * v[:, j]) @ np.log(E))
    spin = np.sqrt(factors @ theta((v @ x) / 2, r) ** 2)
    return spin / spin.sum(), abs(theta(0.0, r)) / abs(theta(0.0, r)).sum()


# From the closed disorder lines along the space and imaginary-time cycles.
M = np.array([[1, 1, 1, 1], [-1, -1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1]])


def normalized(q, gain, drop_zero_mode=False):
    spin, thermal = normalized_blocks(q)
    if drop_zero_mode:
        spin[0] = 0.0
        spin = spin / spin.sum()
    O = (M @ spin).reshape(2, 2)
    Z = (M @ thermal).reshape(2, 2)
    w = np.exp(gain * q["bias"] * np.array([1, -1]))
    if q["seam"] != "free":
        w[np.array([1, -1]) != q["seam"]] = 0.0
    gate = q["parity"]
    obs = O[:, 0].copy()
    den = Z[:, 0].copy()
    if gate:
        obs += gate * O[:, 1]
        den += gate * Z[:, 1]
    base = w @ obs / (w @ den)
    if q["read"] == "response":
        signs = np.array([1, -1])
        return float(
            gain
            * ((w * signs) @ obs / (w @ den) - (w * signs) @ den / (w @ den) * base)
        )
    return float(base)
