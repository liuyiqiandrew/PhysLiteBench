"""Small-Hilbert-space verification in the physical Pauli basis."""
import numpy as np
from scipy.linalg import eigh


def direct(q, N):
    bits = np.arange(2**N)
    Z = np.array([1 - 2 * ((bits >> j) & 1) for j in range(N)])
    P = np.prod(Z, axis=0)
    H = np.diag(-Z.sum(axis=0))
    for j in range(N):
        H[bits, bits ^ (1 << j) ^ (1 << ((j + 1) % N))] -= 1

    def eig(par):
        ix = np.flatnonzero(P == par)
        e, u = eigh(H[np.ix_(ix, ix)].astype(float))
        return ix, e, u

    def pulse(psi, p):
        kind, l, r = p
        i, j = round(l * N), round(r * N)
        if kind == "spin":
            return psi[bits ^ (1 << i) ^ (1 << j)]
        return psi * np.prod(Z[i:j], axis=0)

    plus, _, vectors = eig(1)
    vac = np.zeros(len(bits))
    vac[plus] = vectors[:, 0]
    idx, E, U = eig(1 if q["sector"] == "even" else -1)
    state = np.zeros(len(bits))
    state[idx] = U[:, 0]
    wave = state.astype(complex)
    norm = 1.0
    for a, p in enumerate(q["pulses"]):
        norm *= vac @ pulse(vac, p)
        wave = pulse(wave, p)
        if a < len(q["segments"]):
            t, dt = q["segments"][a]
            coeff = U.T @ wave[idx]
            wave[idx] = U @ (np.exp(-N / 2 * (t + 1j * dt) * (E - E[0])) * coeff)
    return state @ wave / norm
