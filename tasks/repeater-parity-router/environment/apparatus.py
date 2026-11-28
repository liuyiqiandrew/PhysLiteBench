"""Single-request statistics of an elementary-link Bell-pair memory."""
from functools import lru_cache
import math
import numpy as np


def count_weights(mu, nu):
    """Joint weights by total parity and resources: empty, broadband only, one narrow, or narrow plus another."""

    def odd(x):
        return -0.5 * math.expm1(-2 * x)

    zero = math.exp(-(mu + nu))
    out = []
    for parity in (0, 1):
        total = odd(mu + nu) if parity else 1 - odd(mu + nu)
        b = math.exp(-mu) * (odd(nu) if parity else 1 - odd(nu))
        empty = 0.0 if parity else zero
        one = mu * zero if parity else 0.0
        out.append((empty, b - empty, one, max(0.0, total - b - one)))
    return np.array(out)


def paths(links, term):
    adj = {}
    for i, (u, v, *_) in enumerate(links):
        adj.setdefault(u, []).append((v, i))
        adj.setdefault(v, []).append((u, i))

    def visit(v, seen, mask):
        if v == term[1]:
            yield mask
            return
        for w, i in adj[v]:
            if w not in seen:
                yield from visit(w, seen | {w}, mask | 1 << i)

    return list(visit(term[0], {term[0]}, 0))


def parity_patterns(q, a):
    links = q["links"]
    n = len(q["parities"])
    m = len(links)
    counts = np.array([count_weights(a * k, a * l) for u, v, k, l, h in links])
    marg = counts.sum(axis=2)
    masks = np.arange(1 << m)
    odd = (masks[:, None] >> np.arange(m)) & 1
    syn = np.zeros(len(masks), dtype=int)
    for j, (u, v, *_) in enumerate(links):
        syn ^= odd[:, j] * ((1 << u) | (1 << v))
    valid = np.ones(len(masks), dtype=bool)
    for j in range(n):
        if q["parities"][j] is not None:
            valid &= ((syn >> j) & 1) == q["parities"][j]
    odd = odd[valid]
    wp = marg[np.arange(m), odd].prod(axis=1)
    ok = wp > 0
    odd = odd[ok]
    wp = wp[ok]
    return wp / wp.sum(), counts[np.arange(m), odd, :] / marg[np.arange(m), odd, None]


def single(q, a, index):
    """Probability of an individually feasible request and its conditioned XX fringe."""
    links = q["links"]
    m = len(links)
    routes = paths(links, q["requests"][index])
    masks = np.arange(1 << m)
    hit = (masks[:, None] >> np.arange(m)) & 1
    viable = np.array([(masks & route) == route for route in routes])
    num = viable.sum(axis=0)
    good = num > 0
    phase = np.array(
        [math.prod(e[4] for j, e in enumerate(links) if r >> j & 1) for r in routes]
    )
    signal = np.zeros(len(masks))
    signal[good] = (phase @ viable[:, good]) / num[good]
    wp, pc = parity_patterns(q, a)
    popen = pc[:, :, (2 if index == 0 else 1) :].sum(axis=2)
    mass = np.zeros(len(masks))
    for weight, op in zip(wp, popen):
        mass += weight * np.where(hit, op, 1 - op).prod(axis=1)
    p = mass[good].sum()
    return p, mass @ signal / p


class Calibrated:
    def fit(self, rows):
        exposure = np.array([r["input"]["exposure"] for r in rows])
        logp = np.log([r["value"] for r in rows])
        self.activity = float(-(exposure @ logp) / (exposure @ exposure))
        return self

    def predict(self, rows):
        return np.array(
            [
                math.exp(-self.activity * q["exposure"])
                if "exposure" in q
                else self.read(q)
                for q in rows
            ],
            dtype=float,
        )
