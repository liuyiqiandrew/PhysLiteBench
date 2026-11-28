"""Sum the leading replica spin structures selected by the gap records."""
from itertools import product
import numpy as np


def continuum(q):
    if q.get("blank"):
        return np.cos(q["phase"])
    arcs = np.array(q["arcs"])
    m = len(arcs)
    x = arcs.ravel()
    ii, jj = np.triu_indices(len(x), 1)
    dist = 2 * np.sin(np.pi * (x[jj] - x[ii]))
    ref = np.prod((2 * np.sin(np.pi * (arcs[:, 1] - arcs[:, 0]))) ** -0.25)
    rot = q["phase_plates"]
    accepted = [
        g
        for g in product((1, -1), repeat=m)
        if all(np.prod([g[j] for j in mask]) == val for mask, val in q["records"])
    ]
    total = 0.0
    for s in product((1, -1), repeat=m - 1):
        if (*s, int(np.prod(s))) not in accepted:
            continue
        signs = np.cumprod([1, *s]) * np.array(rot)
        for mark in product((0, 1), repeat=m - 1):
            gaps = [1, *[(-1) ** v for v in mark], 1]
            e = []
            for i, sg in enumerate(signs):
                e.extend([sg * gaps[i], -sg * gaps[i + 1]])
            if sum(e) != 0:
                continue
            e = np.array(e)
            coeff = np.prod([s[k] for k, v in enumerate(mark) if v])
            total += coeff * np.prod(dist ** (0.25 * e[ii] * e[jj])) / ref
    return 2 * total / len(accepted)
