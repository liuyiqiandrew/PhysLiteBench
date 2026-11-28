"""Integrate the microscopic parity operation on plane waves and form Slater minors."""
from functools import lru_cache
import warnings
import numpy as np
from scipy.linalg import det, lu_factor, lu_solve, toeplitz, LinAlgWarning


def single_particle(points, L):
    x = np.asarray(points)
    ns = np.arange(1, L)
    v = -np.diff(np.exp(2j * np.pi * ns[:, None] * x), axis=1)[:, ::2].sum(1) / (1j * np.pi * ns)
    zero = 1 - 2 * (x[1::2] - x[::2]).sum()
    row = np.r_[zero, v]
    return toeplitz(row.conj(), row)


@lru_cache(maxsize=1000)
def reference_pair(a, b, N):
    return det(single_particle([a, b], N)).real


def minor_matrix(q, N, direct=False):
    arcs = np.asarray(q["arcs"])
    points = (arcs[:, :2] + arcs[:, 2:] / N).ravel()
    M = N + q["detuning"]
    full = single_particle(points, M + 2)
    # Relative to the final four levels. Every occupied state also contains 0,...,M-3.
    occupied = [(0, 1), (0, 2), (0, 3), (1, 2)]
    omitted = [np.array(sorted(set(range(4)) - set(p))) for p in occupied]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        lu, piv = lu_factor(full)
    # Complementary minors avoid repeating the LU; singular larger matrices use the literal minors.
    if np.min(abs(np.diag(lu))) < 1e-7:
        direct = True
    out = np.zeros((4, 4), complex)
    if direct:
        keep = [np.r_[np.arange(M - 2), M - 2 + np.array(p)] for p in occupied]
        for i, a in enumerate(keep):
            for j, b in enumerate(keep):
                out[i, j] = det(full[np.ix_(a, b)])
    else:
        sign = (-1) ** np.count_nonzero(piv != np.arange(len(piv)))
        d = sign * np.prod(np.diag(lu))
        rhs = np.zeros((M + 2, 4))
        rhs[-4:] = np.eye(4)
        inverse = lu_solve((lu, piv), rhs)[-4:]
        for i, a in enumerate(omitted):
            for j, b in enumerate(omitted):
                out[i, j] = d * (-1) ** int(sum(a) + sum(b)) * np.linalg.det(inverse[np.ix_(b, a)])
    return out / np.prod([reference_pair(a, b, N) for a, b in arcs[:, :2]])


def finite(q, N):
    state = np.array([rad * np.exp(1j * arg) for rad, arg in q["state"]])
    return (np.vdot(state, minor_matrix(q, N) @ state) / np.vdot(state, state)).real


def extrapolate(q, N=192):
    # Four scales remove the first three finite-orbital corrections along a single residue class.
    mult = np.arange(1, 5)
    x = 1 / mult
    return float(np.polynomial.polynomial.polyfit(x, [finite(q, N * int(s)) for s in mult], 3)[0])
