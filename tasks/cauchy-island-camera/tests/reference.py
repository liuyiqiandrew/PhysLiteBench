"""Independent first-exit Green equation for the terminal two-color probability generating function."""
import numpy as np
from scipy.linalg import solve as linear_solve


def solve(q, D, n=110, order=6):
    grid = np.linspace(0, np.pi, n + 2)
    x = np.cos(grid)
    rate = q["rate"]
    d = q["recoil"]
    a, b = q["thresholds"]
    wl, wr = q["efficiencies"]
    split = [
        np.arccos(z)
        for p in (-1.0, 1.0, -1.0 - a, 1.0 + b)
        for shift in (-d, d)
        if -1.0 < (z := p - shift) < 1.0
    ]
    intervals = np.unique(np.r_[grid, split])
    v, w = np.polynomial.legendre.leggauss(14)
    t = (v + 1) / 2
    w = w / 2
    v = np.r_[0.5 * t * t, 1.0 - 0.5 * t * t]
    w = np.r_[w * t, w * t]
    rad = np.diff(intervals)
    phi = (intervals[:-1, None] + rad[:, None] * v).ravel()
    W = (rad[:, None] * w).ravel() * np.sin(phi)
    z = np.cos(phi)
    h = grid[1:-1, None]
    G = np.log(np.abs(np.sin((h + phi) / 2) / np.sin((h - phi) / 2))) / (np.pi * D) * W
    zb = np.exp(2j * np.pi * np.arange(order) / order) * 0.12
    r, s = np.meshgrid(zb, zb, indexing="ij")
    R = np.r_[0.0, 1.0, 0.0, r.ravel()]
    S = np.r_[1.0, 0.0, 0.0, s.ravel()]

    def tail(theta, offset):
        return np.maximum(
            0.0,
            (np.pi - theta) / np.pi
            - 2 / np.pi * np.arctan2(np.sqrt(offset / (2 + offset)), np.tan(theta / 2)),
        )

    terminal = wl * tail(np.pi - grid[1:-1], a)[:, None] * (1 - R) + wr * tail(
        grid[1:-1], b
    )[:, None] * (1 - S)

    def sampler(z):
        angle = np.arccos(np.clip(z, -1.0, 1.0))
        j = np.minimum(n, np.floor(angle * (n + 1) / np.pi).astype(int))
        v = (angle - grid[j]) / (grid[j + 1] - grid[j])
        ext = wl * (z <= -1.0 - a)[..., None] * (1 - R) + wr * (z >= 1.0 + b)[
            ..., None
        ] * (1 - S)

        def sample(values):
            return (1 - v)[..., None] * values[j] + v[..., None] * values[j + 1] + ext

        return sample

    left = sampler(z - d)
    right = sampler(z + d)
    one = sampler(np.asarray(q["launch"]))
    l = sampler(np.asarray(q["launch"] - d))
    rr = sampler(np.asarray(q["launch"] + d))
    coeff = np.zeros((n + 2, len(R)), complex)
    # Resolvent of the within-island lifetime clock, with the exact unmonitored exit source.
    interpolation = np.maximum(
        0.0, 1.0 - np.abs(phi[:, None] - grid[1:-1]) * ((n + 1) / np.pi)
    )
    square = np.eye(n) + rate * G @ interpolation
    H = linear_solve(square, terminal)
    B = linear_solve(square, rate * G)
    for _ in range(140):
        L = left(coeff)
        rhs = right(coeff)
        new = H + B @ (L + rhs - L * rhs)
        err = np.max(np.abs(new - coeff[1:-1]))
        coeff[1:-1] = new
        if err < 2e-12:
            break
    else:
        raise RuntimeError("generating function did not converge")
    if q["source"] == "single":
        u = one(coeff)
    else:
        L = l(coeff)
        right = rr(coeff)
        u = L + right - L * right
    both = np.real(u[0] + u[1] - u[2])
    # Coefficient z^1 w^1 of E[z^red w^blue], with alias error <= 2*(.12)^order.
    exact = np.mean((1 - u[3:].reshape(order, order)) / (r * s)).real
    return np.array(
        [u[0].real, both / u[0].real, exact / both if both > 1e-11 else np.nan]
    )
