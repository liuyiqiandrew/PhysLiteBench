import numpy as np


def explicit(nx, ny, points):
    spins = 1 - 2 * ((np.arange(2 ** (nx * ny))[:, None] >> np.arange(nx * ny)) & 1)
    grid = spins.reshape(-1, ny, nx)
    x = grid * np.roll(grid, -1, axis=2)
    y = grid * np.roll(grid, -1, axis=1)
    base = x[:, :, :-1].sum((1, 2)) + y[:, :-1, :].sum((1, 2))
    ex = x[:, :, -1].sum(1)
    ey = y[:, -1, :].sum(1)
    O = np.prod(np.column_stack([grid[:, j, i] for i, j in points]), axis=1)
    rows = []
    Z = []
    for a, b in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        p = np.exp(0.5 * np.log(1 + np.sqrt(2)) * (base + a * ex + b * ey))
        Z.append(p.sum())
        rows.append(p @ O)
    return np.array(rows) / rows[0], np.array(Z) / Z[0]
