from functools import lru_cache
import numpy as np


@lru_cache(None)
def pairings(start, stop):
    if start == stop:
        return ((),)
    return tuple(
        ((start, j),) + p + q
        for j in range(start + 1, stop, 2)
        for p in pairings(start + 1, j)
        for q in pairings(j + 1, stop)
    )


def folded_average(q, probabilities):
    patterns = pairings(0, len(q["ports"]))
    den, num = 0.0, 0.0
    c, d = q["gate"]
    a, b = q["read"]
    for p, wp in zip(patterns, probabilities[0]):
        for r, wr in zip(patterns, probabilities[1]):
            if wp * wr == 0:
                continue
            labels = list(range(len(q["ports"])))
            for u, v in p + r:
                old, new = labels[u], labels[v]
                labels = [new if x == old else x for x in labels]
            if labels[c] == labels[d]:
                den += wp * wr
                if labels[a] == labels[b]:
                    num += wp * wr
    return num / den


def independent(q, z):
    positions = np.array(q["ports"])
    distributions = []
    for plate in q["plates"]:
        scale = z * plate["ratio"]
        charge = (np.sum(positions, axis=1) % 2) ^ np.array(plate["layer"])
        weights = []
        for p in pairings(0, len(positions)):
            weight = 1.0
            for a, b in p:
                if charge[a] == charge[b]:
                    weight = 0.0
                    break
                dx, dy = positions[a] - positions[b]
                weight /= np.hypot(dx, dy / scale) ** 2
            weights.append(weight)
        distributions.append(np.array(weights) / sum(weights))
    return folded_average(q, distributions)
