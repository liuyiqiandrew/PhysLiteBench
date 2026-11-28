"""Resolve the neutral parity sectors and the four low-energy Slater form factors."""
from functools import lru_cache
from itertools import combinations
import numpy as np

# Traces of powers of the particle coordinates generate the two Schur doublets.
states = [[(1.0, ())], [(1.0, (1,))], [(0.5, (1, 1)), (0.5, (2,))], [(0.5, (1, 1)), (-0.5, (2,))]]


def operator(q):
    arcs = np.asarray(q["arcs"])
    x = arcs[:, :2].ravel()
    shifts = arcs[:, 2:].ravel() + q["detuning"] * x
    m = len(arcs)
    n = 2 * m
    ii, jj = np.triu_indices(n, 1)
    distance = np.log(2 * np.sin(np.pi * (x[jj] - x[ii])))
    ref = np.prod(2 / np.sqrt(2 * np.sin(np.pi * (arcs[:, 1] - arcs[:, 0]))))
    matrix = np.zeros((4, 4), complex)
    circle = np.exp(2j * np.pi * x)
    for plus in combinations(range(n), m):
        e = -np.ones(n)
        e[list(plus)] = 1.0
        weight = np.exp(0.5 * np.dot(e[ii] * e[jj], distance) + 1j * np.pi * (e @ shifts)) / ref
        mu = {k: 0.5 * (e @ (circle**k)) for k in (1, 2)}
        mu.update({-k: -mu[k].conjugate() for k in (1, 2)})

        @lru_cache(maxsize=None)
        def moment(terms):
            if not terms:
                return 1.0
            k, *rest = terms
            val = mu[k] * moment(tuple(rest))
            for i, other in enumerate(rest):
                if other == -k:
                    val += abs(k) * moment(tuple(rest[:i] + rest[i + 1 :]))
            return val

        for i, a in enumerate(states):
            for j, b in enumerate(states):
                matrix[i, j] += weight * sum(
                    c * d * moment(tuple(-k for k in p) + s) for c, p in a for d, s in b
                )
    return matrix


def reading(q, dephase=False):
    if q.get("blank"):
        return np.cos(q["phase"])
    v = np.asarray(q["state"])
    c = v[:, 0] * np.exp(1j * v[:, 1])
    op = operator(q)
    if dephase:
        op = np.diag(np.diag(op))
    return float((np.vdot(c, op @ c) / np.vdot(c, c)).real)
