"""Literal two-level Hilbert states and positive quadrature over remaining Stark couplings."""
from collections import Counter
from itertools import product
from functools import lru_cache
from math import factorial
import numpy as np
from scipy.special import roots_hermitenorm


@lru_cache(maxsize=None)
def gaussian_grid(dim, order):
    if dim == 0:
        return np.empty((1, 0)), np.ones(1)
    x, w = roots_hermitenorm(order)
    points = np.stack(np.meshgrid(*[x] * dim, indexing="ij"), axis=-1).reshape(-1, dim)
    weights = np.prod(
        np.stack(np.meshgrid(*[w / np.sqrt(2 * np.pi)] * dim, indexing="ij"), axis=-1), axis=-1
    ).ravel()
    return points, weights


def prepare(modes, sites):
    # Matrix-free application of two spin raising operators to the product ground state.
    n = len(sites)
    psi = np.zeros(1 << n, complex)
    psi[0] = 1
    for channel in modes:
        new = np.zeros_like(psi)
        for i, p in enumerate(sites):
            amp, angle = channel[p]
            mode = amp * np.exp(1j * angle)
            for state in range(len(psi)):
                if not state & (1 << i):
                    new[state | (1 << i)] += mode * psi[state]
        psi = new
    return psi


def spin_trace(q, sites, means, noise, rates, ratio, order):
    n = len(sites)
    sigs = np.array(list(product((-1, 1), repeat=n)))
    pops = 2**n
    last = 0.0
    hist = [(np.ones(n), 1.0, np.zeros((2, n)))]
    for t, a0, b0, a1, b1 in q["pulses"]:
        cols = np.array([[1, 0], [0, 1], [1, ratio]])[sites]
        areas = np.array([[a0, b0], [a1, b1]]) @ cols.T
        step = np.exp(-2 * rates * (t - last))
        hist = [
            (s, weight * np.prod((1 + s * old * step) / 2), eff + areas * s)
            for old, weight, eff in hist
            for s in sigs
        ]
        last = t
    joint = np.array([prob for _, prob, _ in hist])
    eff = np.array([h for _, _, h in hist])
    field = prepare(q["write"], sites)
    masks = [state for state in range(pops) if state.bit_count() == 2]
    read = [prepare(m, sites).conj() * field for m in q["read"]]
    nodes, weights = gaussian_grid(noise.shape[1], order)
    answer = np.zeros(2, complex)
    for start in range(0, len(nodes), 80):
        couplings = means + nodes[start : start + 80] @ noise.T
        single = np.exp(-1j * eff[:, None, :, :] * couplings[None, :, None, :])
        amplitudes = np.zeros((2, len(hist), len(couplings)), complex)
        for mask in masks:
            i, j = [i for i in range(n) if mask & (1 << i)]
            for arm in (0, 1):
                amplitudes[arm] += read[arm][mask] * single[:, :, arm, i] * single[:, :, arm, j]
        den = abs(amplitudes[0]) ** 2 + abs(amplitudes[1]) ** 2
        cross = amplitudes[1] * amplitudes[0].conj()
        for k, field in enumerate((den, cross)):
            answer[k] += joint @ (field @ weights[start : start + 80])
    return answer


def truth(q, order=19):
    if q.get("blank"):
        return np.cos(q["phase"])
    n = q["count"]
    u, v = q["mix"]
    a, b = q["gate"]
    targets = {(a, b)}
    if q["unsigned_b"]:
        targets.add((a, -b))
    assignments = Counter(tuple(sorted(x)) for x in product(range(3), repeat=n))
    cases = []
    for d in q["domains"]:
        r = d["ratio"]
        pops = np.array(d["pockets"])
        directions = np.array([[1.0, v], [u, 1.0], [1 + u * r, v + r]])
        for index, count in assignments.items():
            sites = np.array(index)
            sig = pops[sites, 1]
            rates = pops[sites, 2]
            matrix = (directions[sites] * sig[:, None]).T
            _, s, vt = np.linalg.svd(matrix, full_matrices=True)
            rank = sum(s > 1e-10)
            weight = (
                d["weight"]
                * np.exp(-sum(pops[:, 0]))
                * np.prod(pops[sites, 0])
                * count
                / factorial(n)
            )
            if rank == 1:
                scale = np.sqrt(2 * np.pi) * np.linalg.norm(matrix, axis=1).max()
            else:
                scale = 2 * np.pi * np.sqrt(np.linalg.det(matrix @ matrix.T))
            for target in targets:
                shift = np.linalg.lstsq(matrix, target, rcond=None)[0]
                if max(abs(matrix @ shift - target)) > 1e-10:
                    continue
                prob = weight * np.exp(-shift @ shift / 2) / scale
                cases.append((rank, prob, sites, sig * shift, sig[:, None] * vt[rank:].T, rates, r))
    best = min(k for k, *_ in cases)
    total = np.zeros(2, complex)
    for k, w, *args in cases:
        if k == best:
            total += w * spin_trace(q, *args, order)
    return float(2 * (np.exp(1j * q["phase"]) * total[1]).real / total[0].real)
