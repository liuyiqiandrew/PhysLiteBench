"""Conditional Gaussian Stark shifts and exact telegraph transition probabilities."""
from itertools import product
from math import factorial
import numpy as np


def modes(polar, sites):
    a = np.asarray(polar)
    return (a[:, :, 0] * np.exp(1j * a[:, :, 1]))[:, sites]


def samples(q):
    """Yield (optical weight, sites, mean F, covariance F, pulse areas, rates).

    Pulse areas have shape (number of sites, number of times, two arms).
    Weights include the microscope's total-count and both spectrometer slits.
    """
    n = q["count"]
    u, v = q["mix"]
    meter = np.array([[1.0, u], [v, 1.0]])
    a, b = q["gate"]
    targets = {(a, b), (a, -b)} if q["unsigned_b"] else {(a, b)}
    kept = []
    for d in q["domains"]:
        r = d["ratio"]
        dirs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, r]])
        pops = np.array(d["pockets"])
        areas = np.asarray(q["pulses"])[:, 1:].reshape(-1, 2, 2)
        for na in range(n + 1):
            for nb in range(n - na + 1):
                occ = np.array([na, nb, n - na - nb])
                sites = np.repeat(np.arange(3), occ)
                p = (
                    d["weight"]
                    * np.exp(-sum(pops[:, 0]))
                    * np.prod([lam**k / factorial(int(k)) for lam, k in zip(pops[:, 0], occ)])
                )
                sig = pops[sites, 1]
                columns = meter @ dirs[sites].T
                U, s, _ = np.linalg.svd(columns * sig, full_matrices=False)
                rank = np.count_nonzero(s > 1e-10)
                basis = U[:, :rank]
                s = s[:rank]
                inverse = (basis / s**2) @ basis.T
                cross = sig[:, None] ** 2 * columns.T
                cov = np.diag(sig**2) - cross @ inverse @ cross.T
                kicks = np.einsum("taj,ij->ita", areas, dirs[sites])
                for target in targets:
                    coord = basis.T @ target
                    if max(abs(basis @ coord - target)) > 1e-10:
                        continue
                    # Equal half widths in the two measured directions.
                    norm = (2 * np.pi) ** (rank / 2) * np.prod(s)
                    if rank == 1:
                        norm *= np.max(abs(basis[:, 0]))
                    w = p * np.exp(-0.5 * np.sum((coord / s) ** 2)) / norm
                    kept.append(
                        (rank, w, sites, cross @ inverse @ target, cov, kicks, pops[sites, 2])
                    )
    degree = min(s[0] for s in kept)
    for rank, *s in kept:
        if rank == degree:
            yield tuple(s)


def histories(times, gamma):
    """All telegraph signs at the pulse times and their probabilities for one site."""
    signs = np.array(list(product((1, -1), repeat=len(times))))
    previous = np.column_stack([np.ones(len(signs)), signs[:, :-1]])
    prob = np.prod(
        (1 + signs * previous * np.exp(-2 * gamma * np.diff(np.r_[0.0, times]))) / 2, axis=1
    )
    return signs, prob


def deterministic(q):
    if q.get("blank"):
        return np.cos(q["phase"])
    times = np.asarray(q["pulses"])[:, 0]
    den = 0.0
    cross = 0j
    for w, sites, mean, cov, kicks, rates in samples(q):
        phases = np.einsum("it,ita,i->ia", np.exp(-2 * rates[:, None] * times), kicks, mean)
        write = modes(q["write"], sites)
        signals = []
        for arm in (0, 1):
            read = modes(q["read"][arm], sites).conj()
            signal = 0j
            for i in range(len(sites)):
                for j in range(i):
                    source = write[0, i] * write[1, j] + write[0, j] * write[1, i]
                    collect = read[0, i] * read[1, j] + read[0, j] * read[1, i]
                    signal += source * collect * np.exp(-1j * (phases[i, arm] + phases[j, arm]))
            signals.append(signal)
        den += w * sum(abs(a) ** 2 for a in signals)
        cross += w * signals[1] * signals[0].conjugate()
    if den == 0:
        return 0.0
    return float(2 * (np.exp(1j * q["phase"]) * cross).real / den)
