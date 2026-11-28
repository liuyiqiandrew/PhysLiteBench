"""Direct integral of the joint disorder transform, retaining its mixed moments."""
from functools import lru_cache
from itertools import combinations
from math import factorial
import numpy as np
from scipy.special import roots_legendre


@lru_cache(None)
def panels(n, edges):
    t, w = roots_legendre(n)
    y = np.concatenate(
        [(a + b) / 2 + (b - a) / 2 * t for a, b in zip(edges, edges[1:])]
    )
    return np.exp(y), np.concatenate(
        [(b - a) * w / 2 for a, b in zip(edges, edges[1:])]
    ) * np.exp(y)


def joined(n, links):
    labs = list(range(n))
    for a, b in links:
        old = labs[b]
        new = labs[a]
        labs = [new if t == old else t for t in labs]
    return tuple(
        tuple(i for i, t in enumerate(labs) if t == k) for k in dict.fromkeys(labs)
    )


def exact(q, tg, n=48, nd=100):
    temps, fields = np.array(q["settings"]).T
    types = np.array(q["preparations"])
    dim = len(types)
    na = int((types == 0).sum())
    nb = dim - na
    s, ds = panels(n, (-57.0, -9.0, 2.0, 12.0))
    x, dx = panels(nd, (-49.0, -8.0, 12.0, 96.0))
    signs = [-1, 1]
    mom = {}
    phi = 0.0
    modes = []
    for m in signs:
        a, b = np.exp(fields[:, None] * m / temps[:, None]) * x[None, :] ** (
            -tg / temps[:, None]
        )
        A = np.exp(-s[:, None] * a)
        B = np.exp(-s[:, None] * b)
        missingA = -np.expm1(-s[:, None] * a)
        missingB = -np.expm1(-s[:, None] * b)
        phi += 0.5 * ((missingA @ dx)[:, None] + (A * dx) @ missingB.T)
        modes.append((a, b, A, B))
    factors = (
        np.outer(ds * s ** (na - 1), ds * s ** (nb - 1))
        * np.exp(-phi)
        / (factorial(na - 1) * factorial(nb - 1))
    )

    def cumulant(indices):
        if indices in mom:
            return mom[indices]
        ix = list(indices)
        u = int((types[ix] == 0).sum())
        v = len(ix) - u
        out = 0.0
        for m, (a, b, A, B) in zip(signs, modes):
            if 0 in ix and q.get("tag") and q["tag"] != m:
                continue
            w = dx * a**u * b**v
            if dim - 1 in ix:
                w = w * np.exp(-q["exposure"] * x)
            out += 0.5 * (A * w) @ B.T
        mom[indices] = out
        return out

    @lru_cache(None)
    def raw(parts):
        if not parts:
            return 1.0
        first, *rest = parts
        val = 0.0
        for h in range(len(rest) + 1):
            for subset in combinations(range(len(rest)), h):
                whole = tuple(sorted(first + sum((rest[j] for j in subset), ())))
                tail = tuple(g for j, g in enumerate(rest) if j not in subset)
                val += cumulant(whole) * raw(tail)
        return val

    def eval_event(links):
        same = [(a, b) for a, b, w in links if w]
        diff = [(a, b) for a, b, w in links if not w]
        total = 0.0
        for h in range(len(diff) + 1):
            for subset in combinations(diff, h):
                groups = joined(dim, same + list(subset))
                total += (-1) ** h * float(np.sum(factors * raw(groups)))
        return total

    d = eval_event(q["gate"])
    return eval_event(q["gate"] + [[dim - 1, q["mark"], True]]) / d, d
