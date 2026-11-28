"""Explicit herald classes, station syndrome and readout of every acceptable pair of simple paths."""
import numpy as np
import math


def simple_paths(edges, n, s, t):
    found = []
    for pattern in range(1, 1 << len(edges)):
        adjacent = [[] for _ in range(n)]
        for k, (u, v, *_) in enumerate(edges):
            if pattern >> k & 1:
                adjacent[u].append(v)
                adjacent[v].append(u)
        if any(
            len(xs) not in ((1,) if i in (s, t) else (0, 2))
            for i, xs in enumerate(adjacent)
        ):
            continue
        used = {s}
        stack = [s]
        while stack:
            v = stack.pop()
            for w in adjacent[v]:
                if w not in used:
                    used.add(w)
                    stack.append(w)
        if used == {v for v, ns in enumerate(adjacent) if ns}:
            found.append(pattern)
    return found


def explicit(q, a):
    edges = q["links"]
    n = len(q["parities"])
    m = len(edges)
    labels = np.arange(4**m, dtype=np.uint32)
    cats = np.array([(labels >> (2 * j)) % 4 for j in range(m)], dtype=np.uint8)
    prior = np.zeros((m, 2, 4))
    for i, (u, v, na, nb, vv) in enumerate(edges):
        for r in range(22):
            p = math.exp(-a * na) * (a * na) ** r / math.factorial(r)
            for s in range(22):
                k = 0 if r + s == 0 else 1 if r == 0 else 2 if r + s == 1 else 3
                prior[i, (r + s) % 2, k] += (
                    p * math.exp(-a * nb) * (a * nb) ** s / math.factorial(s)
                )
    raw = np.zeros(len(labels))
    power = [1 << i for i in range(n)]
    for claim in range(1 << m):
        parity = 0
        for j, (u, v, *_) in enumerate(edges):
            if claim >> j & 1:
                parity ^= power[u] | power[v]
        if any(
            x is not None and (parity >> j & 1) != x
            for j, x in enumerate(q["parities"])
        ):
            continue
        weight = np.ones(len(labels))
        for j in range(m):
            weight *= prior[j, claim >> j & 1, cats[j]]
        raw += weight
    # Explicitly mix equally likely +/- Bell signs per shot with each measured link visibility.
    signs = np.array([[1 - 2 * (s >> j & 1) for j in range(m)] for s in range(1 << m)])
    phase_prob = np.prod((1 + signs * np.array([e[4] for e in edges])) / 2, axis=1)
    red, blue = [simple_paths(edges, n, *p) for p in q["requests"]]
    num = np.zeros(len(labels))
    signal = np.zeros(len(labels))
    ind = np.zeros(len(labels), dtype=bool)
    for r in red:
        for b in blue:
            good = np.ones(len(labels), dtype=bool)
            for i in range(m):
                if r >> i & 1:
                    good &= cats[i] >= 2
                elif b >> i & 1:
                    good &= cats[i] >= 1
            ind |= good
            for i in range(m):
                if r & b & (1 << i):
                    good &= cats[i] == 3
            num[good] += 1
            ss = signs[:, [i for i in range(m) if r >> i & 1]].prod(axis=1) * signs[
                :, [i for i in range(m) if b >> i & 1]
            ].prod(axis=1)
            signal[good] += phase_prob @ ss
    ok = num > 0
    ps = raw[ok].sum()
    return dict(
        delivery=ps / raw.sum(),
        contention=1 - ps / raw[ind].sum(),
        fringe=raw[ok] @ (signal[ok] / num[ok]) / ps,
    )
