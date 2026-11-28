"""Galerkin expansion of the killed |d/dx| generator in weighted Chebyshev U modes."""
import numpy as np
from scipy.linalg import solve
from functools import lru_cache


def centers(q):
    bars = np.array(q["islands"])
    return bars.mean(axis=1), np.diff(bars, axis=1).ravel() / 2


def tag(q, i, z, c, h):
    if q.get("donor") is None:
        return np.ones_like(z)
    mark = q["donor"]
    if mark["island"] != i:
        return np.zeros_like(z)
    s = (1 + (z - c) / h) / 2
    a, b, d = mark["profile"]
    return a * (1 - s) ** 2 + 2 * b * s * (1 - s) + d * s * s


@lru_cache(None)
def modes(n):
    modes = np.arange(1, n + 1)
    nodes, weights = np.polynomial.legendre.leggauss(max(112, 2 * n + 16))
    theta = np.pi * (1 + nodes) / 2
    w = np.pi * weights / 2
    basis = np.sin(modes[:, None] * theta)

    def c(k):
        out = np.zeros(k.shape, dtype=float)
        even = k % 2 == 0
        out[even] = 2 / (1.0 - k[even] ** 2)
        return out

    H = (c(modes[:, None] - modes) - c(modes[:, None] + modes)) / np.pi
    return modes, np.cos(theta), basis * (np.sin(theta) * w), H


def propagate(q, D, n, rate, transfer, source):
    center, h = centers(q)
    count = len(h)
    m, xi, projection, B = modes(n)
    N = count * n
    A = np.zeros((N, N))
    b = np.zeros(N)
    z = [c + r * xi for c, r in zip(center, h)]
    for j, (c, size, x) in enumerate(zip(center, h, z)):
        row = slice(j * n, (j + 1) * n)
        A[row, row] = np.diag(D * m / size) + rate[j] * B
        cross = np.zeros_like(x)
        for k, w in enumerate(z):
            if j == k:
                continue
            col = slice(k * n, (k + 1) * n)
            A[row, col] = (
                -2
                * transfer
                * D
                * h[k]
                / np.pi**2
                * (projection @ (1 / (x[:, None] - w) ** 2) @ projection.T)
            )
            left, right = q["islands"][k]
            cross += D / np.pi * (1 / (left - x) - 1 / (right - x))
        f = (
            rate[j] + (1 - transfer) * cross
            if source == "unconditional"
            else D / np.pi * tag(q, j, x, c, size) / (q["landing"] - x) ** 2
        )
        b[row] = (2 / np.pi) * projection @ f
    C = solve(A, b, check_finite=False)
    x = q["launch"]
    j = next(i for i, (l, r) in enumerate(q["islands"]) if l < x < r)
    value = np.sin(m * np.arccos((x - center[j]) / h[j])) @ C[j * n : (j + 1) * n]
    return 1 - value if source == "unconditional" else value


def reading(q, D, n=72, detached=False):
    zero = [0.0] * len(q["islands"])
    mode = q["observable"]
    unmarked = dict(q, donor=None)
    if mode == "landing":
        return propagate(unmarked, D, n, zero, 1.0, "external")
    tagged = propagate(q, D, n, zero, 1.0, "external")
    if mode == "donor":
        return tagged / propagate(unmarked, D, n, zero, 1.0, "external")
    if detached:
        return propagate(q, D, n, q["rates"], q["relay"], "unconditional")
    return propagate(q, D, n, q["rates"], q["relay"], "external") / tagged
