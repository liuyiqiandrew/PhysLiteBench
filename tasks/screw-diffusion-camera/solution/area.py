"""Pinned stochastic screw area from a harmonic oscillator trace."""
import numpy as np
from scipy.linalg import expm
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import expm_multiply
from numpy.polynomial.hermite import hermgauss
from apparatus import ROT, SCREWS, TRANS
from passive import operators, project


def rotate_area(q, nu, order=6, levels=24, hermite=12):
    e = q["area"]
    f = e["field"]
    a, b = e["axes"]
    freq = abs(f)
    mid = e["middle"]
    finish = np.asarray(e["end"])
    t1, t2 = q["times"]
    k1, k2 = q["wave"]
    pos, spins = operators(order)
    count = (order + 1) ** 2
    gate = []
    for v in q["gates"]:
        root = eye(pos[0].shape[0], format="csr") + project(pos, v)
        gate.append((root @ root)[:count, :count] / 4)

    # The zero-wave gyro closure integral acts only on degrees zero to two.
    angular = [s[:9, :9].toarray() for s in spins]
    bare = sum(r * (j @ j) for r, j in zip(ROT, angular))
    dt = t1 + t2 if mid is None else t1
    offset = finish if mid is None else np.asarray(mid)
    scale = np.sqrt(nu * dt)
    node, wt = hermgauss(hermite)
    den = 0j
    left = gate[0].getrow(0).toarray()[0, :9]
    right = gate[1][:9, 0].toarray()[:, 0]
    for i, x in enumerate(node):
        for j, y in enumerate(node):
            correction = (np.sqrt(ROT[a]) * x * angular[a] + np.sqrt(ROT[b]) * y * angular[b]) / scale
            weight = wt[i] * wt[j] * np.exp(-1j * (x * offset[0] + y * offset[1]) / scale) / np.pi
            den += weight * (left @ expm(nu * t1 * (bare + 2j * correction)) @ right)
    den *= np.exp(offset @ offset / (4 * nu * dt))
    space = eye(count, format="csr")
    osc = eye(levels, format="csr")
    raise_m = diags(np.sqrt(np.arange(1, levels)), 1, shape=(levels, levels), format="csr")
    P = np.sqrt(freq / 2) * (raise_m + raise_m.T)
    Q = 1j * np.sign(f) * np.sqrt(freq / 2) * (raise_m - raise_m.T)
    thermal = diags(-freq * (2 * np.arange(levels) + 1), format="csr")

    def evolve(v, k, t):
        base = -k * k * sum(d * p @ p for d, p in zip(TRANS, pos))
        twist = []
        for j in range(3):
            lever = q["hand"] * SCREWS[j] + np.cross(np.eye(3)[j], q["tag"])
            X = spins[j] + 1j * k * project(pos, lever)
            base += ROT[j] * (X @ X)
            twist.append(X[:count, :count])
        joint = kron(base[:count, :count], osc, format="csr") + kron(space, thermal, format="csr")
        for spin, idx in ((P, a), (Q, b)):
            joint += 2j * np.sqrt(ROT[idx]) * kron(twist[idx], spin, format="csr")
        return expm_multiply(nu * t * joint, v)

    def column(vec):
        return np.kron(vec[:, None], np.eye(levels))

    def pin(tensor, delta, t):
        displacement = expm(-1j * (delta[0] * P + delta[1] * Q).toarray())
        norm = 2 * freq * nu * t * np.exp(delta @ delta / (4 * nu * t))
        return norm * np.einsum("nab,ba->n", tensor.reshape(count, levels, levels), displacement)

    one = np.zeros(count)
    one[0] = 1.0
    last = evolve(column(one), k2, t2)
    if mid is None:
        bridge = kron(gate[1], osc, format="csr") @ last
        answer = pin(evolve(bridge, k1, t1), finish, t1 + t2)
    else:
        first = np.asarray(mid)
        second = finish - first
        bridge = gate[1] @ pin(last, second, t2)
        answer = pin(evolve(column(bridge), k1, t1), first, t1)
        answer *= np.exp(0.5j * f * (first[0] * second[1] - first[1] * second[0]))
    return float((np.exp(-1j * q["phase"]) * (gate[0] @ answer)[0] / den).real)
