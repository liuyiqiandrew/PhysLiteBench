"""Condition on the phase signs at monitored links, then split a Poisson total among their sources."""
from functools import lru_cache
from itertools import product
import math, numpy as np


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


def count_weights(mu, nu):
    pm = np.zeros((2, 4))
    for r in range(22):
        pr = math.exp(-mu) * mu**r / math.factorial(r)
        for b in range(22):
            j = 0 if r + b == 0 else 1 if not r else 2 if r + b == 1 else 3
            pm[(r + b) % 2, j] += pr * math.exp(-nu) * nu**b / math.factorial(b)
    return pm


def integer_compositions(z, k):
    if k == 0:
        if z == 0:
            yield ()
    elif k == 1:
        yield (z,)
    else:
        for m in range(z + 1):
            for tail in integer_compositions(z - m, k - 1):
                yield (m,) + tail


def conditional(q, a):
    links = q["links"]
    watch = q["pilot"]["links"]
    K = q["pilot"]["count"]
    routes = [
        (r, b)
        for r in simple_paths(links, len(q["parities"]), *q["requests"][0])
        for b in simple_paths(links, len(q["parities"]), *q["requests"][1])
    ]
    m = len(links)
    power = np.arange(m)
    sz = 4**m
    labels = np.arange(sz)
    classes = (labels[None, :] >> (2 * power[:, None])) % 4
    ok = np.zeros((len(routes), sz), bool)
    each = np.zeros(sz, dtype=bool)
    for k, (r, b) in enumerate(routes):
        possible = np.ones(sz, dtype=bool)
        for i in range(m):
            if r >> i & 1:
                possible &= classes[i] >= 2
            if b >> i & 1:
                possible &= classes[i] >= 1
        each |= possible
        for i in range(m):
            if r & b & (1 << i):
                possible &= classes[i] == 3
        ok[k] = possible
    viable = ok.sum(axis=0)
    joint = viable > 0
    budget = np.zeros(4)
    u = np.arange(1 << m)
    raw = (u[:, None] >> power) & 1
    syn = np.zeros(len(u), dtype=int)
    for i, (v, w, *_) in enumerate(links):
        syn ^= raw[:, i] * ((1 << v) | (1 << w))
    keep = np.ones(len(u), dtype=bool)
    for j, p in enumerate(q["parities"]):
        if p is not None:
            keep &= (syn >> j & 1) == p
    patterns = raw[keep]
    rates = np.array([e[2:4] for e in links]) * a
    full = [count_weights(*rate) for rate in rates]
    for signs in product((1, -1), repeat=len(watch)):
        vis = np.array([e[4] for e in links])
        pw = 1.0
        neg = []
        for i, sgn in zip(watch, signs):
            pw *= (1 + vis[i] * sgn) / 2
            vis[i] = sgn
            if sgn < 0:
                neg.append(i)
        if not pw:
            continue
        ex = np.array(
            [np.prod([vis[i] for i in range(m) if (r ^ b) >> i & 1]) for r, b in routes]
        )
        fr = np.zeros(sz)
        fr[joint] = (ex @ ok[:, joint]) / viable[joint]
        weights = np.zeros(sz)
        for nums in integer_compositions(K, len(neg)):
            extra = pw
            tables = [*full]
            selected = np.ones(len(patterns), dtype=bool)
            for e, t in zip(neg, nums):
                l, n = rates[e]
                s = l + n
                extra *= math.exp(-s) * s**t / math.factorial(t)
                pi = l / s if s else 0.0
                mass = np.zeros((2, 4))
                p = t % 2
                if t == 0:
                    mass[p, 0] = 1.0
                else:
                    mass[p, 1] = (1 - pi) ** t
                    mass[p, 2 if t == 1 else 3] = 1 - mass[p, 1]
                tables[e] = mass
                selected &= patterns[:, e] == p
            if not extra:
                continue
            for rawbits in patterns[selected]:
                prob = np.full(sz, extra)
                for e, t in enumerate(tables):
                    prob *= t[rawbits[e], classes[e]]
                weights += prob
        budget += [
            weights.sum(),
            weights[each].sum(),
            weights[joint].sum(),
            weights @ fr,
        ]
    norm, sep, success, fringe = budget
    return dict(
        delivery=success / norm, contention=1 - success / sep, fringe=fringe / success
    )
