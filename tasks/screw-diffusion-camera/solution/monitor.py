"""Continuously observed spin-3/2 in two body-mounted complementary doublets."""
import numpy as np
from scipy.sparse import eye, kron, bmat
from scipy.sparse.linalg import expm_multiply
from apparatus import ROT, SCREWS, TRANS
from passive import operators, project
from polarimeter import bands, spin_axes as J

I = np.eye(2)


def spin(v):
    x, y, z = v
    return np.array([[z, x - 1j * y], [x + 1j * y, -z]])


def transfer(K):
    return np.kron(K, K.conj())


def monitored(q, nu, order=6):
    s = q["monitor"]
    maxcount = s["changes"]
    count = maxcount + 1
    bases = [bands(f) for f in s["filters"]]
    initial = s["start"]
    rho = (I + spin(s["initial"])) / 2
    analyzer = I if s["analyzer"] is None else (I + spin(s["analyzer"])) / 2
    pos, ang = operators(order)
    n = (order + 1) ** 2
    space = eye(n, format="csr")
    pair = eye(4, format="csr")
    gates = []
    for v in q["gates"]:
        w = eye(pos[0].shape[0], format="csr") + project(pos, v)
        gates.append((w @ w)[:n, :n] / 4)

    def generator(k, B):
        spatial = -k * k * sum(d * (p @ p) for p, d in zip(pos, TRANS))
        twists = []
        for j in range(3):
            tag = q["hand"] * SCREWS[j] + np.cross(np.eye(3)[j], q["tag"])
            X = ang[j] + 1j * k * project(pos, tag)
            spatial += ROT[j] * (X @ X)
            twists.append(X[:n, :n])
        orbital = kron(spatial[:n, :n], pair, format="csr")
        blocks = [[None] * count for _ in range(count)]
        for c in range(count):
            a = (initial + c) % 2
            within = orbital.copy()
            changes = np.zeros((4, 4), complex)
            for j, r in enumerate(ROT):
                H = J[j]
                W = B[a].conj().T @ H @ B[a]
                outside = B[1 - a].conj().T @ H @ B[a]
                D = B[a].conj().T @ H @ H @ B[a]
                S = 1j * W
                first = np.kron(S, I) + np.kron(I, S.conj())
                second = -np.kron(D, I) - np.kron(I, D.conj()) + 2 * transfer(S)
                within += r * (
                    2 * kron(twists[j], first.T, format="csr") + kron(space, second.T, format="csr")
                )
                changes += 2 * r * transfer(outside).T
            blocks[c][c] = within
            if c < maxcount:
                blocks[c][c + 1] = kron(space, changes, format="csr")
        return bmat(blocks, format="csr")

    middle = [[None] * count for _ in range(count)]
    for c in range(count):
        i = (initial + c) % 2
        for h in [0, 1]:
            if c + h >= count:
                continue
            j = (i + h) % 2
            overlap = bases[1][j].conj().T @ bases[0][i]
            middle[c][c + h] = kron(gates[1], transfer(overlap).T, format="csr")
    middle = bmat(middle, format="csr")
    start = np.zeros((count, n, 4), complex)
    start[0] = gates[0].getrow(0).toarray().T * rho.reshape(1, 4)

    def signal(wave, effect):
        tail = np.zeros((count, n, 4), complex)
        tail[-1, 0] = effect.T.ravel()
        value = tail.reshape(-1)
        for i in (1, 0):
            value = expm_multiply(nu * q["times"][i] * generator(wave[i], bases[i]), value)
            if i == 1:
                value = middle @ value
        return np.dot(start.ravel(), value)

    value = signal(q["wave"], analyzer)
    if s["read"] == "selected":
        norm = signal([0.0, 0.0], I)
    else:
        free = sum(r * (a @ a) for r, a in zip(ROT, ang))[:n, :n]
        unit = np.zeros(n)
        unit[0] = 1.0
        norm = (gates[0] @ expm_multiply(nu * q["times"][0] * free, gates[1] @ unit))[0]
    return float((np.exp(-1j * q["phase"]) * value / norm).real)
