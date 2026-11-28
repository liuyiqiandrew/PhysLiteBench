"""Count selected phase-pilot photons jointly with feasible path resources and station syndrome."""
from collections import defaultdict
from functools import lru_cache
import math
from apparatus import paths, count_weights


def marked_weights(q, a, signature, detach=False, saturate=False):
    K = q["pilot"]["count"]
    watched = set(q["pilot"]["links"])
    out = []
    for i, (_, _, h, f, v) in enumerate(q["links"]):
        mu, nu = a * h, a * f
        pm = count_weights(mu, nu)
        mark = bool(signature >> i & 1)
        terms = []
        for p in (0, 1):
            for c in range(4):
                base = pm[p, c]
                signfactor = v if mark else 1.0
                if i not in watched:
                    if base:
                        terms.append((p, c, 0, base * signfactor))
                    continue
                trait = 1 if detach else -1 if mark else 1
                common = signfactor if detach else 1.0
                if base:
                    terms.append((p, c, 0, base * (1 + v) * 0.5 * common))
                if saturate:
                    t = int(c != 0)
                    neg = base * (1 - v) * 0.5 * trait * common
                    if t <= K and neg:
                        terms.append((p, c, t, neg))
                    continue
                for t in range(p, K + 1, 2):
                    zero = math.exp(-mu - nu) / math.factorial(t)
                    exact = (
                        zero
                        if t == 0 and c == 0
                        else zero * nu**t
                        if t > 0 and c == 1
                        else zero * mu
                        if t == 1 and c == 2
                        else zero * ((mu + nu) ** t - nu**t)
                        if t >= 2 and c == 3
                        else 0.0
                    )
                    if exact:
                        terms.append((p, c, t, exact * (1 - v) * 0.5 * trait * common))
        out.append(terms)
    return out


def law(
    q, a, duplicate=False, fringe=False, detach=False, saturate=False, denominator=False
):
    links = q["links"]
    K = q["pilot"]["count"]
    par = q["parities"]
    p1, p2 = [paths(links, t) for t in q["requests"]]
    choices = [(r, b) for r in p1 for b in p2]
    full = (1 << len(choices)) - 1
    if denominator:
        choices = [(0, 0)]
        full = 1
    masks = []
    for j in range(len(links)):
        edge = 1 << j
        masks.append(
            [
                sum(1 << i for i, (r, b) in enumerate(choices) if not (r | b) & edge),
                sum(1 << i for i, (r, b) in enumerate(choices) if not r & edge),
                full
                if duplicate
                else sum(
                    1 << i for i, (r, b) in enumerate(choices) if not (r & b) & edge
                ),
                full,
            ]
        )
    last = {v: i for i, (u, w, *_) in enumerate(links) for v in (u, w)}
    signatures = {0: full}
    if fringe:
        signatures = {}
        for i, (r, b) in enumerate(choices):
            signatures[r ^ b] = signatures.get(r ^ b, 0) | (1 << i)
    result = 0.0
    for mask, candidates in signatures.items():
        terms = marked_weights(q, a, mask, detach, saturate)
        dp = {(0, 0, full): 1.0}
        for i, (u, v, *_) in enumerate(links):
            ends = (1 << u) | (1 << v)
            forget = sum(1 << w for w in (u, v) if last[w] == i)
            know = sum(1 << w for w in (u, v) if last[w] == i and par[w] is not None)
            target = sum(1 << w for w in (u, v) if last[w] == i and par[w] == 1)
            nd = defaultdict(float)
            for (image, k, live), mass in dp.items():
                for p, c, t, pr in terms[i]:
                    raw = image ^ (ends if p else 0)
                    valid = live & masks[i][c]
                    if raw & know != target or t + k > K or not valid:
                        continue
                    nd[raw & ~forget, t + k, valid] += mass * pr
            dp = nd
        result += sum(
            mass
            * (
                (options & candidates).bit_count() / options.bit_count()
                if fringe
                else 1.0
            )
            for (_, k, options), mass in dp.items()
            if k == K
        )
    return result


def value(q, a, detach=False, saturate=False):
    ps = law(q, a, saturate=saturate)
    if q["observable"] == "delivery":
        norm = law(q, a, denominator=True, saturate=saturate)
        return ps / norm if norm else 0.0
    if q["observable"] == "contention":
        sep = law(q, a, duplicate=True, saturate=saturate)
        return 1 - ps / sep if sep else 0.0
    return law(q, a, fringe=True, detach=detach, saturate=saturate) / ps if ps else 0.0
