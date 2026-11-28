"""Critical square lattice, dual domain walls with the microscopic winding sectors separated."""
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu


@lru_cache(None)
def edges(nx, ny):
    y, x = np.indices((ny, nx))
    base = (y * nx + x) * 4
    row = []
    col = []
    phase = []
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    for d in range(4):
        next_base = 4 * (((y + dy[d]) % ny) * nx + (x + dx[d]) % nx)
        for turn in [-1, 0, 1]:
            row.extend((base + d).ravel())
            col.extend((next_base + ((d + turn) % 4)).ravel())
            phase.extend([np.exp(1j * turn * np.pi / 4)] * (nx * ny))
    return np.array(row), np.array(col), np.array(phase)


def defects(nx, ny, points):
    s = np.ones((ny, nx, 4))
    for (ax, ay), (bx, by) in zip(points[::2], points[1::2]):
        x0, x1 = sorted([ax, bx])
        y0, y1 = sorted([ay, by])
        s[ay - 1, x0:x1, 1] *= -1
        s[ay, x0:x1, 3] *= -1
        s[y0:y1, bx - 1, 0] *= -1
        s[y0:y1, bx, 2] *= -1
    return s


def fermions(nx, ny, points):
    defects0 = defects(nx, ny, points)
    row, col, phase = edges(nx, ny)
    logs = []
    for sx, sy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
        if sx == 1 and sy == 1 and not len(points):
            logs.append(-np.inf)
            continue
        v = defects0.copy()
        v[:, -1, 0] *= sx
        v[:, 0, 2] *= sx
        v[-1, :, 1] *= sy
        v[0, :, 3] *= sy
        T = coo_matrix(
            ((np.sqrt(2) - 1) * phase * v.ravel()[row], (row, col)),
            shape=(4 * nx * ny, 4 * nx * ny),
        ).tocsc()
        L = splu(eye(T.shape[0], format="csc") - T, permc_spec="COLAMD")
        logs.append(0.5 * np.log(np.abs(L.U.diagonal())).sum())
    logs = np.array(logs)
    v = np.exp(logs - max(logs))
    return v / v.sum()
