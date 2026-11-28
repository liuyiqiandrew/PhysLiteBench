import numpy as np

from apparatus import contacts, polarizations


def pfaffian(a):
    """Skew elimination with a pivot in each remaining first row."""
    a = a.copy()
    result = 1. + 0j
    for i in range(0, len(a), 2):
        p = i + 1 + np.argmax(np.abs(a[i, i+1:]))
        if a[i, p] == 0:
            return 0j
        if p != i + 1:
            a[[p, i+1], :] = a[[i+1, p], :]
            a[:, [p, i+1]] = a[:, [i+1, p]]
            result = -result
        pivot = a[i, i+1]
        result *= pivot
        x, y = a[i, i+2:], a[i+1, i+2:]
        a[i+2:, i+2:] += (np.outer(y, x) - np.outer(x, y)) / pivot
    return result


def ordered_average(cov, sequence, pairs, parity_vectors):
    vectors = pairs[[i for a, b, _ in sequence for i in (a, b)]]
    theta = np.array([angle for _, _, angle in sequence])
    c = np.cos(theta/2)
    s = -np.sin(theta/2).astype(complex)
    if len(parity_vectors):
        vectors = np.vstack((vectors, parity_vectors))
        count = len(parity_vectors) // 2
        c = np.r_[c, np.zeros(count)]
        s = np.r_[s, np.full(count, -1j)]
    if not len(c):
        return 1.
    weighted = vectors.astype(complex)
    weighted[::2] *= s[:, None]
    upper = np.triu(weighted @ cov @ weighted.T, 1)
    even = np.arange(0, len(upper), 2)
    upper[even, even+1] += c
    return pfaffian(upper - upper.T)


def fringe(q, parity):
    d = 2*q["n"]
    pair = contacts(q)
    t = polarizations(q)
    cov = np.eye(d, dtype=complex)
    j = np.arange(0,d,2)
    cov[j, j+1] = 1j*t
    cov[j+1, j] = -1j*t
    empty = np.empty((0,d))
    modes = np.eye(d)
    paths = [[(a,b,turn + sign*coupling) for a,b,turn,coupling in reversed(arm)]
             for arm in q["arms"] for sign in (1,-1)]
    weights = np.array([1, q["outcome"]*np.exp(-1j*q["meter_phase"])])
    gram = np.eye(4, dtype=complex)
    for i in range(4):
        for k in range(i+1,4):
            seq = [(a,b,-angle) for a,b,angle in reversed(paths[i])] + paths[k]
            average = ordered_average(cov, seq, pair, empty)
            if parity:
                with_parity = ordered_average(cov, seq, pair, modes)
                average = (average + parity*with_parity) / (1 + parity*np.prod(t))
            gram[i,k], gram[k,i] = average, average.conjugate()
    den = (weights.conj() @ (gram[:2,:2] + gram[2:,2:]) @ weights).real
    if den < 1e-14:
        return 0.
    cross = weights.conj() @ gram[:2,2:] @ weights
    return float(2*(np.exp(-1j*q["phase"])*cross).real/den)
