"""Critical spin Hamiltonian in its even spin-flip sector on a finite ring."""
from functools import lru_cache
import numpy as np


def pfaffian(matrix):
    A = matrix.copy()
    value = 1.0 + 0j
    for k in range(0, len(A) - 1, 2):
        p = k + 1 + np.argmax(abs(A[k, k + 1 :]))
        if A[k, p] == 0:
            return 0j
        if p != k + 1:
            A[[p, k + 1]] = A[[k + 1, p]]
            A[:, [p, k + 1]] = A[:, [k + 1, p]]
            value = -value
        a = A[k, k + 1]
        value *= a
        x = A[k, k + 2 :]
        y = A[k + 1, k + 2 :]
        A[k + 2 :, k + 2 :] += (np.outer(y, x) - np.outer(x, y)) / a
    return value


@lru_cache(256)
def kernel(N, gap, odd):
    offset = 0.0 if odd else 0.5
    modes = np.arange(-N, 0) + offset
    k = np.pi / N * modes
    pad = np.zeros(2 * N, complex)
    pad[:N] = np.exp(2 * N * np.sin(k) * gap)
    if odd:
        pad[0] = 0.0
    d = np.arange(-2 * N, 2 * N)
    coefficients = np.fft.ifft(pad) * 2
    return np.exp(1j * np.pi / N * (-N + offset) * d) * coefficients[d % (2 * N)]


def dipole(N, p, offset=0):
    kind, l, r = p
    a, b = round(N * l), round(N * r)
    # X_i X_j is the ordered product (-i b_k a_(k+1)) on its bonds;
    # the dipole of dual spins is the product (-i a_k b_k) on its sites.
    shift = (kind == "spin") + 2 * offset * (kind == "string")
    return np.arange(2 * a + shift, 2 * b + shift)


def wick(N, indices, times, odd=False):
    sites = np.concatenate(indices)
    groups = np.repeat(np.arange(len(indices)), list(map(len, indices)))
    A = np.empty((len(sites), len(sites)), complex)
    for a, g in enumerate(indices):
        for b in range(a, len(indices)):
            h = indices[b]
            gap = times[a] - times[b]
            shifted = g[:, None] - h[None, :] + 2 * N
            part = kernel(N, gap, odd)[shifted]
            if odd:
                # The uniform even/odd-sublattice zero modes have occupied
                # combined fermion in the physical odd spin-flip sector.
                phase = np.where(g % 2, 1j, 1.0)
                other = np.where(h % 2, -1j, 1.0)
                part += phase[:, None] * other[None, :] / N
            A[np.ix_(groups == a, groups == b)] = part
    A = np.triu(A, 1)
    A -= A.T
    return pfaffian(A) * ((-1j) ** (len(sites) // 2))


def value(q, N, offset=0):
    indices = [dipole(N, p, offset) for p in q["pulses"]]
    reg = np.array(q["segments"])
    times = np.r_[0.0, np.cumsum(reg[:, 0] + 1j * reg[:, 1])]
    norm = np.prod([wick(N, [i], [0.0]) for i in indices])
    return wick(N, indices[::-1], times[::-1], q.get("sector", "even") == "odd") / norm
