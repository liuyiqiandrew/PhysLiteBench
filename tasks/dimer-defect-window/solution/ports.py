"""Boundary strands on independently covered bipartite sheets."""
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


@lru_cache(None)
def patterns(start, stop):
    if start == stop:
        return ((),)
    return tuple(
        ((start, partner),) + inside + outside
        for partner in range(start + 1, stop, 2)
        for inside in patterns(start + 1, partner)
        for outside in patterns(partner + 1, stop)
    )


@lru_cache(None)
def incidence(n):
    links = patterns(0, n)
    signs = np.array([[any(a == j for a, b in p) for j in range(n)] for p in links])
    matrix = np.array(
        [[all(s[a] != s[b] for a, b in p) for p in links] for s in signs], dtype=float
    )
    return signs, np.linalg.inv(matrix)


def response(w, h, ports, activity):
    vertices = [(x, y) for x in range(w) for y in range(h)]
    blacks = {p: i for i, p in enumerate(p for p in vertices if sum(p) % 2 == 0)}
    whites = {p: i for i, p in enumerate(p for p in vertices if sum(p) % 2)}
    ii, jj, vv = [], [], []
    for (x, y), i in blacks.items():
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if (x + dx, y + dy) in whites:
                ii.append(i)
                jj.append(whites[x + dx, y + dy])
                vv.append(1.0 if dx else activity * (-1) ** x)
    n = len(blacks)
    factor = splu(coo_matrix((vv, (ii, jj)), shape=(n, n)).tocsc())
    bi = [blacks[p] for p in ports if p in blacks]
    wi = [whites[p] for p in ports if p in whites]
    rhs = np.zeros((n, len(bi)))
    rhs[bi, np.arange(len(bi))] = 1
    return factor.solve(rhs)[wi, :]


def distribution(q, z, plate):
    ports = tuple(map(tuple, q["ports"]))
    dark = np.array([sum(p) % 2 == 0 for p in ports])
    signs, invert = incidence(len(ports))
    coupling = response(q["w"], q["h"], ports, z * plate["ratio"])
    sums = []
    for sign in signs:
        remove = sign ^ dark

        def minor(r):
            return abs(np.linalg.det(coupling[np.ix_(r[~dark], r[dark])]))

        sums.append(minor(remove) * minor(~remove))
    weights = invert @ np.array(sums)
    charge = np.array(plate["layer"], dtype=bool) ^ dark
    legal = [all(charge[a] != charge[b] for a, b in p) for p in patterns(0, len(ports))]
    weights = np.maximum(weights, 0) * legal
    return weights / sum(weights)


@lru_cache(None)
def connections(n):
    links = patterns(0, n)
    groups = np.empty((len(links), len(links), n), dtype=int)
    for i, p in enumerate(links):
        for j, q in enumerate(links):
            components = list(range(n))
            for a, b in p + q:
                v, u = components[a], components[b]
                components = [v if x == u else x for x in components]
            groups[i, j, :] = components
    return groups


def folded(q, z):
    left, right = [distribution(q, z, p) for p in q["plates"]]
    law = np.outer(left, right)
    group = connections(len(q["ports"]))
    a, b = q["read"]
    c, d = q["gate"]
    keep = group[:, :, c] == group[:, :, d]
    positive = group[:, :, a] == group[:, :, b]
    return np.sum(law[keep & positive]) / np.sum(law[keep])
