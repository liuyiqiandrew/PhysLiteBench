"""Average interference between every allowed hard-core pair of excited defects."""
from itertools import combinations, product
from functools import lru_cache
import numpy as np
from spectrometer import samples, modes, histories


@lru_cache(maxsize=None)
def indices(states, n):
    return np.array(list(product(range(states), repeat=n))).T


def reading(q, distinguish_pairs=False):
    if q.get("blank"):
        return np.cos(q["phase"])
    times = np.array([p[0] for p in q["pulses"]])
    den = 0.0
    cross = 0j
    for w, sites, mean, cov, kicks, rates in samples(q):
        n = len(sites)
        pairs = list(combinations(range(n), 2))
        write = modes(q["write"], sites)

        def exchange(arr, i, j):
            return arr[0, i] * arr[1, j] + arr[0, j] * arr[1, i]

        weights = []
        for read in q["read"]:
            field = modes(read, sites).conj()
            weights.append(
                np.array([exchange(field, i, j) * exchange(write, i, j) for i, j in pairs])
            )
        state, p = zip(*(histories(times, g) for g in rates))
        coeff = np.einsum("igt,ita->iga", state, kicks)

        def metric(a, b):
            value = 0j
            memo = {}
            for i, pair in enumerate(pairs):
                for j, other in enumerate(pairs):
                    if distinguish_pairs and j != i:
                        continue
                    terms = tuple(sorted(set(pair) | set(other)))
                    counts = tuple((int(sites[k]), k in pair, k in other) for k in terms)
                    if counts not in memo:
                        ix = indices(len(p[0]), len(terms))
                        h = np.array(
                            [
                                int(k in pair) * coeff[k, ix[l], a]
                                - int(k in other) * coeff[k, ix[l], b]
                                for l, k in enumerate(terms)
                            ]
                        )
                        prob = np.prod([p[k][ix[l]] for l, k in enumerate(terms)], axis=0)
                        t = list(terms)
                        var = np.sum(h * (cov[np.ix_(t, t)] @ h), axis=0)
                        memo[counts] = prob @ np.exp(-0.5 * var - 1j * (mean[t] @ h))
                    value += weights[a][i] * weights[b][j].conjugate() * memo[counts]
            return value

        den += w * (metric(0, 0) + metric(1, 1)).real
        cross += w * metric(1, 0)
    return float(2 * (np.exp(1j * q["phase"]) * cross).real / den)
