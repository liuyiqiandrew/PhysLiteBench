"""Direct symmetrization on a frequency grid, with coherent propagation in rail space."""
from itertools import permutations, product
from math import factorial
import numpy as np
from scipy.special import roots_hermitenorm


def interferometer(n, rows):
    u = np.eye(n, dtype=complex)
    for i, j, t, p in rows:
        a, b = u[[i, j]].copy()
        u[i] = np.cos(t) * a - np.exp(-1j * p) * np.sin(t) * b
        u[j] = np.exp(1j * p) * np.sin(t) * a + np.cos(t) * b
    return u


def channels(q):
    state = q["spectral"]["preparations"]
    n = len(state[0]["mean"])
    spec = q["spectral"]
    w = complex(*spec.get("weight", [0, 0]))
    m = complex(*spec.get("marker", [0, 0]))

    def emit(s, x):
        d = x - np.array(s["mean"])
        c = np.array(s["cov"])
        v = np.einsum("...i,ij,...j->...", d, np.linalg.inv(c), d)
        phase = 0.5 * np.einsum("...i,ij,...j->...", x, s["chirp"], x) - x @ np.array(
            s["delay"]
        )
        return (
            np.exp(-0.25 * v + 1j * phase)
            / (np.linalg.det(c) * (2 * np.pi) ** n) ** 0.25
        )

    def psis(x):
        a = emit(state[0], x)
        if len(state) == 1:
            return [a]
        b = w * emit(state[1], x)
        return [a + m * b, np.sqrt(max(0.0, 1 - abs(m) ** 2)) * b]

    return n, psis


def grid(n, order):
    # The tabulated weights below integrate dω, not a probability density.
    x, w = roots_hermitenorm(order)
    x = x * 0.91
    w = 0.91 * w * np.exp(0.5 * (x / 0.91) ** 2)
    return x, w


def reading(q, order=34):
    n, packets = channels(q)
    x, w = grid(n, order)
    spatial = np.array(list(product(range(n), repeat=n)))
    perms = list(permutations(range(n)))
    incoming = interferometer(n, q["before"])
    out = interferometer(n, q["after"])
    kron = np.array([[1.0]])
    for _ in range(n):
        kron = np.kron(kron, out)
    ids = np.arange(n)
    bank = np.isin(spatial, q["bank"]).sum(axis=1)
    observable = np.cos(q["lambda"] * (bank - len(q["bank"])) - q["phase"])
    gate = q["gate"]
    keep = (
        np.ones(len(spatial))
        if gate is None
        else ((-1) ** np.isin(spatial, gate["rails"]).sum(axis=1) == gate["parity"])
    )
    so = spatial == q["spectral"]["plate"][0]
    delay, chirp = q["spectral"]["plate"][1:]
    mass = contrast = norm = 0.0
    coefs = [
        np.prod(incoming[spatial, np.asarray(p)], axis=1) / np.sqrt(factorial(n))
        for p in perms
    ]
    invs = [np.argsort(p) for p in perms]
    size = order**n
    for first in range(0, size, 1400):
        inds = np.stack(
            np.unravel_index(np.arange(first, min(first + 1400, size)), (order,) * n),
            axis=1,
        )
        z = x[inds]
        wt = np.prod(w[inds], axis=1)
        fs = [packets(z[:, iv]) for iv in invs]
        norm += sum(wt @ abs(a) ** 2 for a in fs[0])
        plate = np.exp(1j * ((-delay * z + chirp * z * z / 2) @ so.T))
        for i in range(len(fs[0])):
            psi = sum(v[i][:, None] * u for u, v in zip(coefs, fs))
            after = (psi * plate * keep) @ kron.T
            prob = wt @ (abs(after) ** 2)
            mass += sum(prob)
            contrast += prob @ observable
    return float(contrast / mass), float(mass / norm)
