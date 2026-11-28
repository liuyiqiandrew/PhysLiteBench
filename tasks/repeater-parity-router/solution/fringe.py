"""Read one uniformly selected feasible ordered path pair per successful herald record."""
from collections import defaultdict
from apparatus import paths, parity_patterns
import math


def ensemble(q, a, fresh=False):
    links = q["links"]
    choices = [
        (r, b)
        for r in paths(links, q["requests"][0])
        for b in paths(links, q["requests"][1])
    ]
    full = (1 << len(choices)) - 1
    allowed = []
    fringes = []
    for j in range(len(links)):
        e = 1 << j
        allowed.append(
            [
                sum(1 << k for k, (r, b) in enumerate(choices) if not (r | b) & e),
                sum(1 << k for k, (r, b) in enumerate(choices) if not r & e),
                sum(1 << k for k, (r, b) in enumerate(choices) if not (r & b) & e),
                full,
            ]
        )
    for r, b in choices:
        if fresh:
            val = math.prod(
                edge[4] ** (((r >> j) & 1) + ((b >> j) & 1))
                for j, edge in enumerate(links)
            )
        else:
            val = math.prod(edge[4] for j, edge in enumerate(links) if (r ^ b) >> j & 1)
        fringes.append(val)
    post, loads = parity_patterns(q, a)
    weighted = total = pooled = paths_total = 0.0
    for prior, pc in zip(post, loads):
        states = {full: prior}
        for masks, prob in zip(allowed, pc):
            nxt = defaultdict(float)
            for options, mass in states.items():
                for keep, p in zip(masks, prob):
                    viable = options & keep
                    if viable and p:
                        nxt[viable] += mass * p
            states = nxt
        for options, mass in states.items():
            count = options.bit_count()
            signal = sum(v for k, v in enumerate(fringes) if options >> k & 1)
            weighted += mass * signal / count
            total += mass
            pooled += mass * signal
            paths_total += mass * count
    return weighted / total, pooled / paths_total
