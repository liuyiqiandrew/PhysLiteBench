"""Sum herald records while retaining all feasible boundary parities of the two routes."""
from collections import defaultdict
from functools import lru_cache


from apparatus import count_weights


def weights(q, scale, duplicate=False):
    links = q["links"]
    parities = q["parities"]
    N = len(parities)
    term = [sum(1 << v for v in route) for route in q["requests"]]
    last = {v: k for k, (u, w, *_) in enumerate(links) for v in (u, w)}
    states = {(0, frozenset([0])): 1.0}
    for k, (u, v, m, n, *extra) in enumerate(links):
        endpoints = (1 << u) | (1 << v)
        red = endpoints
        blue = endpoints << N
        retire = sum(1 << j for j in (u, v) if last[j] == k)
        observed = sum(
            1 << j for j in (u, v) if last[j] == k and parities[j] is not None
        )
        raw_target = sum(1 << j for j in (u, v) if last[j] == k and parities[j] == 1)
        bothret = retire | (retire << N)
        want = (term[0] & retire) | ((term[1] & retire) << N)
        modes = (
            (0,),
            (0, blue),
            (0, red, blue, red | blue) if duplicate else (0, red, blue),
            (0, red, blue, red | blue),
        )

        @lru_cache(None)
        def advance(h, j):
            values = (f ^ p for f in h for p in modes[j])
            return frozenset(x & ~bothret for x in values if x & bothret == want)

        pm = count_weights(scale * m, scale * n)
        nxt = defaultdict(float)
        for (raw, history), weight in states.items():
            for parity in (0, 1):
                after = raw ^ (endpoints if parity else 0)
                if after & observed != raw_target:
                    continue
                old = after & ~retire
                for j, p in enumerate(pm[parity]):
                    if p > 0:
                        nxt[old, advance(history, j)] += weight * p
        states = nxt
    return sum(weight for (_, h), weight in states.items() if h), sum(states.values())
