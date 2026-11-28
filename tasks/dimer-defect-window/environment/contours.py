"""Evaluate the double-dimer contour generating function on a rectangular sheet."""
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def partition_ratio(q, z, common, single):
    w, h = q["w"], q["h"]
    # Slits from two plaquette centers to the top boundary. Distinct columns
    # keep the slits disjoint. The counterclockwise holonomies are A and B.
    if single:
        a = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
        b = np.array([[1.0, 0.0], [2 * common - 2, 1.0]], dtype=complex)
    else:
        a = np.diag([1j, -1j])
        b = np.array([[-1j * common, 1], [common**2 - 1, 1j * common]])
    sites = [(x, y) for y in range(h) for x in range(w)]
    blacks = {p: i for i, p in enumerate(p for p in sites if sum(p) % 2 == 0)}
    whites = {p: i for i, p in enumerate(p for p in sites if sum(p) % 2 == 1)}
    links = dict(zip((tuple(p) for p in q["marks"]), (a, b)))
    rr, cc, vv = [], [], []
    scalar_rows, scalar_cols, scalar = [], [], []
    for (x, y), i in blacks.items():
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            other = (x + dx, y + dy)
            if other not in whites:
                continue
            j = whites[other]
            weight = 1.0 if dx else (-1) ** x * q["ratio"] * z
            mat = np.eye(2, dtype=complex)
            for (px, py), m in links.items():
                if dx and min(x, x + dx) == px and y > py:
                    mat = m if dx > 0 else np.array([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]])
            rr.extend([2 * i, 2 * i, 2 * i + 1, 2 * i + 1])
            cc.extend([2 * j, 2 * j + 1, 2 * j, 2 * j + 1])
            vv.extend((weight * mat).ravel())
            scalar_rows.append(i)
            scalar_cols.append(j)
            scalar.append(weight)
    n = len(blacks)

    def det_data(rows, cols, vals, n):
        k = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
        try:
            lu = splu(k)
        except RuntimeError:
            return 0, -math.inf
        d = lu.U.diagonal()
        phase = np.prod(d / abs(d))
        for perm in (lu.perm_r, lu.perm_c):
            seen = set()
            for i in range(n):
                if i in seen:
                    continue
                j = i
                size = 0
                while j not in seen:
                    seen.add(j)
                    j = perm[j]
                    size += 1
                if size % 2 == 0:
                    phase = -phase
        return phase, np.log(abs(d)).sum()

    phase, log = det_data(rr, cc, vv, 2 * n)
    _, base = det_data(scalar_rows, scalar_cols, scalar, n)
    return phase * math.exp(log - 2 * base)


def reading(q, z):
    n = q["count"]
    allow = q["allow_separate"]
    if n == 0:
        return float(partition_ratio(q, z, 0.0, allow).real)
    degree = min(min(x + 1, y + 1, q["w"] - 1 - x, q["h"] - 1 - y) for x, y in q["marks"])
    k = degree + 1
    if n >= k:
        return 0.0
    angles = 2 * np.pi * np.arange(k) / k
    return float(
        np.mean(
            [partition_ratio(q, z, np.exp(1j * a), allow) * np.exp(-1j * n * a) for a in angles]
        ).real
    )
