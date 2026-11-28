"""Critical spin blocks on a doubly periodic film."""
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


M = np.array([[1, 1, 1, 1], [-1, -1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1]])


def blocks(q):
    xy = np.array(q["marks"])
    r = q["aspect"]
    x = xy[:, 0] + 1j * r * xy[:, 1]
    n = len(x)
    signs = []
    for plus in combinations(range(1, n), n // 2 - 1):
        v = -np.ones(n)
        v[[0, *plus]] = 1
        signs.append(v)
    v = np.array(signs)
    i, j = np.triu_indices(n, 1)
    pair = np.abs(theta(x[i] - x[j], r)[:, 0])
    weights = np.exp(0.5 * (v[:, i] * v[:, j]) @ np.log(pair))
    spin = np.sqrt(weights @ abs(theta((v @ x) / 2, r)) ** 2)
    Z = abs(theta(0.0, r))
    return M @ spin / spin.sum(), M @ Z / Z.sum()
