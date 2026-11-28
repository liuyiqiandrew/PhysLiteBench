"""Exact one-interval Green and exit kernels coupled at the first landing on another island."""
from functools import lru_cache
import numpy as np
from scipy.linalg import solve


@lru_cache(None)
def mesh(n):
    th = np.linspace(0, np.pi, n + 2)
    g, w = np.polynomial.legendre.leggauss(18)
    t = (g + 1) / 2
    w = w / 2
    # Split every angular panel in half; quadratic endpoint maps resolve the Green logarithm at either end.
    v = np.r_[0.5 * t * t, 1 - 0.5 * t * t]
    dw = np.r_[w * t, w * t] * np.pi / (n + 1)
    phi = (th[:-1, None] + (np.pi / (n + 1)) * v).ravel()
    W = np.tile(dw, n + 1)
    local = np.tile(v, n + 1)
    return th, phi, W, local


def calculate(q, D, n, tagged, clocked):
    bars = np.asarray(q["islands"])
    C = bars.mean(axis=1)
    H = (bars[:, 1] - bars[:, 0]) / 2
    th, phi, W, u = mesh(n)
    N = n * len(C)
    M = np.eye(N)
    rhs = np.zeros(N)

    def row(i, theta):
        width = H[i]
        x = C[i] + width * np.cos(theta)
        xz = C[i] + width * np.cos(phi)
        size = len(x)
        green = np.log(
            np.abs(
                np.sin((theta[:, None] + phi) / 2) / np.sin((theta[:, None] - phi) / 2)
            )
        ) / (np.pi * D)
        wg = green * (W * np.sin(phi) * width)
        selection = 1.0
        if tagged and q.get("donor"):
            mk = q["donor"]
            a, b, c = mk["profile"]
            s = (1 + np.cos(phi)) / 2
            selection = (
                (a * (1 - s) ** 2 + 2 * b * s * (1 - s) + c * s**2)
                if mk["island"] == i
                else 0.0
            )
        source = wg @ (D * selection / (np.pi * (q["landing"] - xz) ** 2))
        trans = np.zeros((size, N))
        for j in range(len(C)):
            if i == j:
                weights = -(q["rates"][i] if clocked else 0.0) * wg
            else:
                y = C[j] + H[j] * np.cos(phi)
                p = (
                    width
                    * np.sin(theta[:, None])
                    / (
                        np.pi
                        * np.sqrt(abs((y - bars[i, 0]) * (y - bars[i, 1])))
                        * abs(y - x[:, None])
                    )
                )
                weights = (
                    p * (W * np.sin(phi) * H[j]) * (q["relay"] if clocked else 1.0)
                )
            a = weights.reshape(size, n + 1, -1)
            interp = (a * u.reshape(n + 1, -1)[None, :, :]).sum(axis=2)[:, :-1] + (
                a * (1 - u.reshape(n + 1, -1))[None, :, :]
            ).sum(axis=2)[:, 1:]
            trans[:, j * n : (j + 1) * n] = interp
        return source, trans

    for i in range(len(C)):
        source, trans = row(i, th[1:-1])
        seg = slice(i * n, (i + 1) * n)
        rhs[seg] = source
        M[seg] -= trans
    f = solve(M, rhs, check_finite=False)
    x = q["launch"]
    j = next(i for i, (l, r) in enumerate(bars) if l < x < r)
    # Interpolation at the launch gives a second-order weak approximation independent of the generator basis.
    return np.interp(
        np.arccos((x - C[j]) / H[j]), th, np.r_[0.0, f[j * n : (j + 1) * n], 0.0]
    )


def reading(q, D, n=110):
    if q["observable"] == "landing":
        return calculate(q, D, n, False, False)
    if q["observable"] == "donor":
        return calculate(q, D, n, True, False) / calculate(q, D, n, False, False)
    return calculate(q, D, n, True, True) / calculate(q, D, n, True, False)
