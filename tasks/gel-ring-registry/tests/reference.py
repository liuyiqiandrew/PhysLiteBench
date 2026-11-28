from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu


@lru_cache(None)
def tree(n):
    names = [()]
    lookup = {(): 0}
    src = []
    dest = []
    letters = []
    outer = []
    directions = []
    for i, t in enumerate(names):
        for d in range(4):
            u = t[:-1] if t and (t[-1] ^ 1) == d else t + (d,)
            if len(u) > n:
                outer.append(i)
                directions.append(d)
                continue
            if u not in lookup:
                lookup[u] = len(names)
                names.append(u)
            src.append(i)
            dest.append(lookup[u])
            letters.append(d)
    rim = np.array([i for i, a in enumerate(names) if len(a) == n])
    sel = [i for i in rim if names[i][0] != (names[i][-1] ^ 1)]
    return (
        names,
        np.array(src),
        np.array(dest),
        np.array(letters),
        np.array(outer),
        np.array(directions),
        rim,
        np.array(sel),
    )


def branches(p, z):
    v = np.array(p["routes"], float)
    s = 1 / (1 + z * p["feed"])
    q = v / v.sum() * (1 - s)
    g = np.ones(4)
    for i in range(90):
        excursion = q * g[np.arange(4) ^ 1] * q[np.arange(4) ^ 1]
        g = 1 / (1 - excursion.sum() + excursion)
    return q, s, g * q


def truth(q, z, depth=None):
    names, src, dst, d, outer, dirs, rim, sel = tree(
        q["length"] if depth is None else depth
    )
    N = len(names)
    if depth is not None:
        sel = np.array(
            [
                i
                for i, w in enumerate(names)
                if len(w) == q["length"] and w[0] != (w[-1] ^ 1)
            ]
        )
    ops = []
    hits = []
    for k, p in enumerate([q["trunk"], *q["arms"]]):
        route, s, f = branches(p, z)
        hits.append(f)
        P = coo_matrix(
            (
                np.r_[route[d], route[dirs] * f[dirs ^ 1] * (depth is None)],
                (np.r_[src, outer], np.r_[dst, outer]),
            ),
            shape=(N, N),
        ).tocsc()
        op = splu(eye(N, format="csc") - P)
        if k == 0:
            source = np.zeros(N)
            source[0] = s
            ops.append(op.solve(source, trans="T"))
        else:
            obs = np.zeros((N, len(sel)))
            obs[sel, np.arange(len(sel))] = s
            ops.append(op.solve(obs))
    # If the trunk lies beyond this shell, both daughters independently have to reach the shell.
    tail = hits[0] * hits[1][np.arange(4) ^ 1] * hits[2][np.arange(4) ^ 1]
    layer = tail.copy()
    summed = layer.copy()
    for _ in range(65):
        layer = tail * (layer.sum() - layer[np.arange(4) ^ 1])
        summed += layer
    idx = [a[-1] ^ 1 for a in (names[i] for i in rim)]
    if depth is None:
        ops[0][rim] *= 1 + summed.sum() - summed[idx]
    groups = {}
    selected = [names[i] for i in sel]
    for i, w in enumerate(selected):
        key = min(w[j:] + w[:j] for j in range(len(w)))
        groups.setdefault(key, []).append(i)
    num = den = 0.0
    for key, II in groups.items():
        II = np.asarray(II)
        for iv in II:
            left = [
                i
                for i in II
                if selected[i][0] in q["gate"] and (not q["shifted"] or iv != i)
            ]
            if not left:
                continue
            weight = ops[0] @ (ops[1][:, left].sum(axis=1) * ops[2][:, iv])
            if "probe" in q:
                val = int(selected[iv][-1] in q["probe"])
            else:
                x, y = q["flux"]
                angs = [x, -x, y, -y]
                val = 0.5 + 0.5 * np.cos(q["phase"] + sum(angs[d] for d in key))
            num += weight * val
            den += weight
    return float(num / den), float(den)
