"""Reconstruct planar strand probabilities from positive boundary-deletion sums."""
from functools import lru_cache
from itertools import combinations
import numpy as np


@lru_cache(None)
def diagrams(n):
    if n == 0:
        return ((),)
    result = []
    for middle in range(1, n, 2):
        for a in diagrams(middle - 1):
            for b in diagrams(n - middle - 1):
                result.append(
                    (middle,)
                    + tuple(i + 1 for i in a)
                    + (0,)
                    + tuple(i + middle + 1 for i in b)
                )
    return tuple(result)


@lru_cache(None)
def measurement_matrix(n):
    coords = np.arange(n)
    colorings = []
    matrix = []
    for subset in combinations(range(1, n), n // 2 - 1):
        tag = np.isin(coords, (0,) + subset)
        colorings.append(tag)
        matrix.append(
            [all(tag[i] != tag[j] for i, j in enumerate(p)) for p in diagrams(n)]
        )
    return np.array(colorings), np.array(matrix, dtype=bool)


def boundary_law(q, z, p):
    ports = list(map(tuple, q["ports"]))
    k = len(ports)
    colorings, incidence = measurement_matrix(k)
    vertices = [(x, y) for y in range(q["h"]) for x in range(q["w"])]
    b, w = [[v for v in vertices if sum(v) % 2 == color] for color in (0, 1)]
    ix = {v: i for i, v in enumerate(w)}
    mat = np.zeros((len(b), len(w)), complex)
    for row, (x, y) in enumerate(b):
        for v, a in [
            ((x + 1, y), 1),
            ((x - 1, y), 1),
            ((x, y + 1), 1j * z * p["ratio"]),
            ((x, y - 1), 1j * z * p["ratio"]),
        ]:
            if v in ix:
                mat[row, ix[v]] = a
    black = np.array([sum(v) % 2 == 0 for v in ports])

    # Direct deleted determinants / common scale. All the deletion sites are
    # on the exterior face, leaving every interior Kasteleyn face intact.
    def logz(holes):
        active = {v for v, t in zip(ports, holes) if t}
        keepb, keepw = [np.array([v not in active for v in side]) for side in (b, w)]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.linalg.slogdet(mat[np.ix_(keepb, keepw)])[1]

    logs = []
    for state in colorings:
        removed = state == black
        logs.append(logz(removed) + logz(~removed))
    # A rare boundary assignment can be over 30 orders of magnitude lighter
    # than an ordinary one. Each compatible assignment bounds the mass of a
    # diagram. Use its smallest such bound to keep the least-squares equations
    # accurate relative to the preparation that will actually be observed.
    logs = np.asarray(logs)
    bounds = np.min(np.where(incidence, logs[:, None], np.inf), axis=0)
    viable = np.isfinite(bounds)
    supported = np.isfinite(logs)
    # Exponentiate only entries that appear in the equation. Their exponent
    # is nonpositive by construction.
    active = incidence[np.ix_(supported, viable)]
    ratios = bounds[viable] - logs[supported, None]
    matrix = np.exp(np.where(active, ratios, -np.inf))
    relative = np.linalg.lstsq(matrix, np.ones(sum(supported)), rcond=None)[0]
    masses = np.zeros(len(bounds))
    masses[viable] = np.maximum(relative, 0) * np.exp(
        bounds[viable] - max(bounds[viable])
    )
    charges = black ^ np.array(p["layer"], dtype=bool)
    for i, d in enumerate(diagrams(k)):
        if any(charges[a] == charges[b] for a, b in enumerate(d)):
            masses[i] = 0
    return masses / sum(masses)


def connected(p, q, a, b):
    step, endpoint = p, a
    while True:
        endpoint = step[endpoint]
        if endpoint == b:
            return True
        if endpoint == a:
            return False
        step = q if step is p else p


def folded(q, z):
    ps, qs = [boundary_law(q, z, p) for p in q["plates"]]
    den, num = 0.0, 0.0
    for i, p in enumerate(diagrams(len(q["ports"]))):
        for j, r in enumerate(diagrams(len(q["ports"]))):
            if ps[i] * qs[j] > 0 and connected(p, r, *q["gate"]):
                den += ps[i] * qs[j]
                if connected(p, r, *q["read"]):
                    num += ps[i] * qs[j]
    return num / den
