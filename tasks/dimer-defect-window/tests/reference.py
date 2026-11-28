"""A second cut placement and determinant lemma for nesting-count targets."""
import math
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy.linalg import det
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def reading(q, z):
    w, h = q["w"], q["h"]
    points = sorted([tuple(m) for m in q["marks"]])
    sites = [(x, y) for x in range(w) for y in range(h)]
    black = {p: i for i, p in enumerate(p for p in sites if sum(p) % 2 == 0)}
    white = {p: i for i, p in enumerate(p for p in sites if sum(p) % 2 == 1)}
    row, col, entries, changed = [], [], [], []
    for (x, y), i in black.items():
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            if (x + dx, y + dy) not in white:
                continue
            j = white[x + dx, y + dy]
            weight = 1.0 if dx else (1j * z * q["ratio"])
            row.append(i)
            col.append(j)
            entries.append(weight)
            for side, (px, py) in enumerate(points):
                if (
                    dy
                    and min(y, y + dy) == py
                    and ((side == 0 and x <= px) or (side == 1 and x > px))
                ):
                    direction = dy if side else -dy
                    changed.append((i, j, weight, side, direction))
    n = len(black)
    k = coo_matrix((entries, (row, col)), shape=(n, n)).tocsc()
    inv_columns = splu(k).solve(np.eye(n, dtype=complex)[:, [e[0] for e in changed]])
    selected = inv_columns[[e[1] for e in changed], :]

    def generating(t):
        if q["allow_separate"]:
            matrices = [np.array([[1, 2], [0, 1]]), np.array([[1, 0], [t - 1, 1]])]
        else:
            b = -t + 1j * math.sqrt(max(0, 1 - t * t))
            matrices = [np.array([[0, -1], [1, 0]]), np.array([[0, -b], [1 / b, 0]])]
        ids = np.eye(2)
        blocks = np.kron(selected, ids)
        for edge, (_, _, weight, side, sense) in enumerate(changed):
            m = matrices[side]
            if sense == -1:
                m = np.linalg.inv(m)
            sl = slice(2 * edge, 2 * edge + 2)
            blocks[sl, :] = (weight * (m - ids)) @ blocks[sl, :]
        return det(np.eye(len(blocks)) + blocks).real

    degree = min(min(x + 1, y + 1, w - 1 - x, h - 1 - y) for x, y in points)
    if q["count"] > degree:
        return 0.0
    if q["count"] == 0:
        return float(generating(0.0))
    nodes = Chebyshev.basis(degree + 1).roots()
    vals = [generating(float(t)) for t in nodes]
    polynomial = Chebyshev.fit(nodes, vals, degree, domain=[-1.0, 1.0]).convert(kind=Polynomial)
    return float(polynomial.coef[q["count"]])


TRUE_PARAMETER = 0.937


def brute(q, z):
    w, h = q["w"], q["h"]
    n = w * h
    v = z * q["ratio"]

    def coverings(mask, bonds, weight):
        if not mask:
            yield frozenset(bonds), weight
            return
        a = (mask & -mask).bit_length() - 1
        for j, weight1 in [(a + 1, 1.0), (a + w, v)]:
            if (j == a + 1 and a % w == w - 1) or j >= n or not mask & (1 << j):
                continue
            yield from coverings(mask ^ (1 << a) ^ (1 << j), bonds + [(a, j)], weight * weight1)

    ds = list(coverings((1 << n) - 1, [], 1.0))
    denom = sum(t for _, t in ds) ** 2
    val = 0.0
    for i, (m1, wa) in enumerate(ds):
        for k, (m2, wb) in enumerate(ds[: i + 1]):
            edges = m1 ^ m2
            graph = {}
            for a, b in edges:
                graph.setdefault(a, []).append(b)
                graph.setdefault(b, []).append(a)
            counts = [0, 0]
            while graph:
                p = next(iter(graph))
                vertex = p
                last = None
                cuts = [0, 0]
                while True:
                    neighbors = graph.pop(vertex)
                    new = neighbors[0] if neighbors[0] != last else neighbors[1]
                    if abs(new - vertex) == 1:
                        for t, (mx, my) in enumerate(q["marks"]):
                            cuts[t] += int(min(new, vertex) % w == mx and vertex // w > my)
                    last, vertex = vertex, new
                    if vertex == p:
                        break
                enclosed = sum(i % 2 for i in cuts)
                if enclosed:
                    counts[enclosed - 1] += 1
            if counts[1] == q["count"] and (q["allow_separate"] or counts[0] == 0):
                val += (1 if k == i else 2) * wa * wb
    return val / denom
