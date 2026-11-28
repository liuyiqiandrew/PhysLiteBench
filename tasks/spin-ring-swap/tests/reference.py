"""Finite-ring replica determinants and a direct spin-basis contraction."""
from itertools import combinations, product
import numpy as np
from scipy.linalg import det


def fits(bits, records):
    return all(np.prod(np.array(bits)[ix]) == sign for ix, sign in records)


def regions(q, N):
    arcs = (np.array(q["arcs"]) * N).astype(int)
    A = [np.arange(a, b) for a, b in arcs]
    G = [np.arange(b, c) for (a, b), (c, d) in zip(arcs[:-1], arcs[1:])]
    G.append(np.r_[np.arange(arcs[-1, 1], N), np.arange(arcs[0, 0])])
    return A, G


def probdet(fields):
    N = len(fields)
    M = N // 2
    ix = np.arange(M)
    mat = np.fft.fft(fields) / N
    return abs(det(mat[ix[:, None] - ix[None, :]])) ** 2


def finite(q, N):
    A, G = regions(q, N)
    m = len(A)
    rot = q["phase_plates"]
    norm = 1.0
    for arc in A:
        v = np.ones(N, complex)
        v[arc] = 1j
        norm *= probdet(v)
    allowed = [g for g in product((1, -1), repeat=m) if fits(g, q["records"])]
    prob = 0.0
    for bits in product((0, 1), repeat=m):
        coef = sum(np.prod([g[j] for j, at in enumerate(bits) if at]) for g in allowed) / 2**m
        if coef:
            v = np.ones(N)
            for j, at in enumerate(bits):
                if at:
                    v[G[j]] = -1.0
            prob += coef * probdet(v)
    weighted = 0.0
    for s in product((1, -1), repeat=m - 1):
        if not fits((*s, int(np.prod(s))), q["records"]):
            continue
        levels = np.cumprod([1, *s]) * np.array(rot)
        for mark in product((0, 1), repeat=m - 1):
            v = np.ones(N, complex)
            for block, sg in zip(A, levels):
                v[block] = 1j * sg
            for k, b in enumerate(mark):
                if b:
                    v[G[k]] = -1.0
            weighted += np.prod([s[k] for k, b in enumerate(mark) if b]) * probdet(v) / 2 ** (m - 1)
    return weighted / prob / norm


def wave(q, N):
    A, G = regions(q, N)
    array = np.concatenate(A)
    other = np.setdiff1d(np.arange(N), array)
    phi = np.zeros((2 ** len(array), 2 ** len(other)))
    for part in combinations(range(N), N // 2):
        vec = np.zeros(N, int)
        vec[list(part)] = 1
        ix = vec[array] @ (2 ** np.arange(len(array)))
        j = vec[other] @ (2 ** np.arange(len(other)))
        delta = np.subtract.outer(part, part)
        u = delta[np.tril_indices(len(part), -1)]
        phi[ix, j] = np.sin(np.pi * u / N).prod()
    phi /= np.linalg.norm(phi)
    x = np.arange(phi.shape[1])[:, None] >> np.arange(len(other)) & 1
    signs = np.array([(-1) ** x[:, np.isin(other, g)].sum(1) for g in G]).T
    rho = {
        tuple(p): phi[:, np.all(signs == p, axis=1)] @ phi[:, np.all(signs == p, axis=1)].T
        for p in np.unique(signs, axis=0)
    }
    num = den = 0.0
    z = np.ones(len(array), int)
    for arc, v in zip(A, q["phase_plates"]):
        z[np.isin(array, arc)] = v
    phases = np.prod(
        z[None, :] ** (np.arange(phi.shape[0])[:, None] >> np.arange(len(array)) & 1), axis=1
    )
    for pa, ra in rho.items():
        for pb, rb in rho.items():
            if not fits(tuple(a * b for a, b in zip(pa, pb)), q["records"]):
                continue
            den += np.trace(ra) * np.trace(rb)
            num += np.sum(ra * rb * phases * phases[:, None])
    # Independently obtain each contiguous purity by forming the exact one-block reduced density matrix.
    normal = 1
    for block in A:
        indices = [list(array).index(k) for k in block]
        a = phi.reshape([2] * len(array) + [phi.shape[1]])
        axes = [len(array) - 1 - j for j in indices]
        ps = np.transpose(a, axes + [i for i in range(len(array) + 1) if i not in axes]).reshape(
            1 << len(axes), -1
        )
        normal *= np.linalg.norm(ps @ ps.T) ** 2
    return num / den / normal


def extrapolate(q, base=128):
    mult = np.arange(1, 5)
    return np.polynomial.polynomial.polyfit(1 / mult, [finite(q, base * int(n)) for n in mult], 3)[
        0
    ]
