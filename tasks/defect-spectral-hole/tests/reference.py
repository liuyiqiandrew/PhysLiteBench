"""Finite occupancies, explicit labels of the emitting defects, and conditional Gaussian fields."""
from functools import lru_cache
from itertools import product
import numpy as np
from scipy.special import gammaln, roots_legendre, xlogy


@lru_cache(maxsize=None)
def histograms(n, k):
    if k == 1:
        return [(n,)]
    return [(i,) + v for i in range(n + 1) for v in histograms(n - i, k - 1)]


def kernel(pop, direction, kicks, horizon, microphone, hits, x, ceiling):
    lam, sigma, gamma = pop
    eta, red = microphone
    rate = eta * gamma
    x = np.atleast_1d(x)
    out = np.zeros((len(x), 2), complex)
    signatures = []
    for states in product((1, -1), repeat=len(kicks)):
        weight = 1.0
        last = 0.0
        old = 1
        coeff = []
        for s, (t, a, b) in zip(states, kicks):
            tau = gamma * (t - last) - rate * (min(t, horizon) - min(last, horizon))
            weight *= (1 + s * old * np.exp(-2 * tau)) / 2
            coeff.append(s * (a * direction[0] + b * direction[1]))
            last = t
            old = s
        signatures.append((weight, coeff))
    prob = np.array([p for p, v in signatures])
    coeff = np.array([v for p, v in signatures])
    times = np.array([t for t, a, b in kicks])
    initial = coeff.sum(1)

    def forced(block):
        signs = (-1) ** np.array([sum(t > f for f, c in block) for t in times])
        return coeff @ signs

    partitions = [[hits]] if hits else [[]]
    if len(hits) == 2:
        partitions.append([[hits[0]], [hits[1]]])
    for blocks in partitions:
        if any(len({c for t, c in block}) > 1 for block in blocks):
            continue
        confidence = np.prod(
            [red if block[0][1] == "R" else 1 - red for block in blocks]
        ) * rate ** len(hits)
        draws = list(product(range(len(prob)), repeat=len(blocks)))
        event_w = np.array([np.prod([prob[i] for i in choice]) for choice in draws])
        tagged = np.array(
            [[forced(block)[i] for i, block in zip(choice, blocks)] for choice in draws]
        )
        t1 = tagged.sum(1)
        t2 = (tagged**2).sum(1)
        k = len(blocks)
        for n in range(max(1, k), ceiling + 1):
            m = n - k
            freq = np.asarray(histograms(m, len(prob)))
            ordinary = np.exp(xlogy(freq, prob).sum(1) - gammaln(freq + 1).sum(1))
            # n!/(n-k)! labeled sources cancel the same factorials in the Poisson/multinomial weights.
            w = (
                np.exp(-lam - n * rate * horizon)
                * lam**n
                * confidence
                * (ordinary[:, None] * event_w)
            )
            avg = (freq @ initial)[:, None] + t1
            spread = (freq @ (initial**2))[:, None] + t2 - avg**2 / n
            g = np.exp(-0.5 * x * x / (n * sigma * sigma)) / np.sqrt(2 * np.pi * n * sigma * sigma)
            out[:, 0] += g * w.sum()
            out[:, 1] += g * (
                np.exp((-1j / n) * x[:, None] * avg.ravel())
                @ (w * np.exp(-0.5 * sigma * sigma * spread)).ravel()
            )
    return out


def truth(q, ceiling=15, degree=180):
    if q.get("blank"):
        return np.cos(q["phase"])
    a, b = q["gate"]
    u, v = q["mix"]
    meter = np.array([[1.0, u], [v, 1.0]])
    readings = np.unique([[a, b], [a, -b]] if q["unsigned_b"] else [[a, b]], axis=0)
    choices = []
    for domid, domain in enumerate(q["domains"]):
        r = domain["ratio"]
        directions = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, r]])
        columns = directions @ meter.T
        for assignment in product(range(3), repeat=len(q["clicks"])):
            by_cell = tuple(
                tuple(tuple(q["clicks"][i]) for i, k in enumerate(assignment) if k == j)
                for j in range(3)
            )
            if any(
                any(
                    q["microphones"][j][0] == 0
                    or (q["microphones"][j][1] if color == "R" else 1 - q["microphones"][j][1]) == 0
                    for t, color in cell
                )
                for j, cell in enumerate(by_cell)
            ):
                continue
            for target in readings:
                for mask in product((False, True), repeat=3):
                    if any(cell and not flag for cell, flag in zip(by_cell, mask)):
                        continue
                    used = np.flatnonzero(mask)
                    matrix = columns[used].T
                    if len(used):
                        best, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
                    else:
                        best = np.empty(0)
                        rank = 0
                    if np.max(abs(matrix @ best - target)) < 1e-10:
                        choices.append((rank, domid, directions, columns, used, best, by_cell))
    rank = min(item[0] for item in choices)
    results = np.zeros(2, complex)
    cache = {}
    for degree_of_support, domid, directions, columns, used, best, by_cell in choices:
        if degree_of_support != rank:
            continue
        dom = q["domains"][domid]
        pockets = dom["pockets"]
        w = dom["weight"] * np.exp(-sum(p[0] for i, p in enumerate(pockets) if i not in used))

        def density(i, x):
            key = (domid, i, by_cell[i], np.atleast_1d(x).tobytes())
            if key not in cache:
                cache[key] = kernel(
                    pockets[i],
                    directions[i],
                    q["kicks"],
                    q["listen_until"],
                    q["microphones"][i],
                    by_cell[i],
                    x,
                    ceiling,
                )
            return cache[key]

        if rank == 0:
            results += w
        elif rank == 1:
            i = used[0]
            results += w * density(i, best[0])[0] / np.max(abs(columns[i]))
        elif len(used) == 2:
            i, j = used
            results += (
                w
                * (density(i, best[0]) * density(j, best[1]))[0]
                / abs(np.linalg.det(columns[used]))
            )
        else:
            _, _, vt = np.linalg.svd(columns.T, full_matrices=True)
            z, weights = roots_legendre(degree)
            coords = best + 18.0 * z[:, None] * vt[-1]
            area = 18.0 / np.sqrt(np.linalg.det(columns.T @ columns))
            both = np.prod([density(i, coords[:, i]) for i in used], axis=0)
            results += w * area * (weights @ both)
    return float((np.exp(-1j * q["phase"]) * results[1]).real / results[0].real)
