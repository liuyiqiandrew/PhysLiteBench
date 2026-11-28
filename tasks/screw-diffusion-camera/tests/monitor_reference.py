"""Advance signed finite material screws, observing the complete spin matrix after each kick."""
import numpy as np
from scipy.linalg import expm
from scipy.sparse import bmat, csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply
from reference import grid, rates, generator

levels = np.arange(1.5, -2, -1.0)
up = np.zeros((4, 4))
for j, m in enumerate(levels):
    if j:
        up[j - 1, j] = np.sqrt(3.75 - m * (m + 1))
axes = np.array([0.5 * (up + up.T), -0.5j * (up - up.T), np.diag(levels)])


def active(record):
    v = record["turn"]
    rotate = expm(-1j * sum(a * r for a, r in zip(axes, v)))
    first = [1, 2] if record["band"] == "central" else [0, 1]
    second = [j for j in range(4) if j not in first]
    return [rotate[:, first], rotate[:, second]]


def bloch(v):
    x, y, z = v
    return np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]]) / 2


def ref_monitor(q, nu, order=7, step=0.002):
    u, w, Y, adj, neighbors = grid(order, step)
    N = Y.shape[1]
    s = q["monitor"]
    lim = s["changes"]
    total = lim + 1
    tag = u @ q["tag"]
    k1, k2 = q["wave"]
    t1, t2 = q["times"]
    I = np.eye(2)
    gates = [(1 + u @ g) ** 2 / 4 for g in q["gates"]]
    intervals = [active(z) for z in s["filters"]]

    def channel(K):
        # Explicit action on each density-matrix unit in column storage.
        mats = []
        for a in range(2):
            for b in range(2):
                mats.append(np.outer(K[:, a], K[:, b].conj()).reshape(4))
        return np.array(mats)

    def evolution(k, B):
        size = 4 * N
        transl = csr_matrix(adj @ ((-k * k * (u**2 @ [0.18, 0.27, 0.36]))[:, None] * Y))
        center = kron(transl, eye(4), format="csr") - 2 * sum(rates) / step**2 * eye(
            size, format="csr"
        )
        own = [center.copy(), center.copy()]
        jump = [csr_matrix((size, size), dtype=complex) for _ in range(2)]
        for j, dirs in enumerate(neighbors):
            for (basis, distance), angle in zip(dirs, [step, -step]):
                turn = expm(1j * angle * axes[j])
                move = csr_matrix(adj @ (np.exp(1j * k * q["hand"] * distance)[:, None] * basis))
                for old in [0, 1]:
                    for new, block in [(old, own), (1 - old, jump)]:
                        change = B[new].conj().T @ turn @ B[old]
                        block[old] += rates[j] / step**2 * kron(move, channel(change), format="csr")
        for block in own + jump:
            block.data[abs(block.data) < 5e-8] = 0
            block.eliminate_zeros()
        blocks = [[None] * total for _ in range(total)]
        for n in range(total):
            b = (s["start"] + n) % 2
            blocks[n][n] = own[b]
            if n < lim:
                blocks[n][n + 1] = jump[b]
        return bmat(blocks, format="csr")

    def numerator(waves, effect):
        a, b = waves
        tail = np.zeros((total, N, 4), complex)
        tail[-1] = (adj @ np.exp(1j * b * tag))[:, None] * effect.T.ravel()
        v = expm_multiply(nu * t2 * evolution(b, intervals[1]), tail.ravel()).reshape(total, N, 4)
        weight = adj @ ((gates[1] * np.exp(1j * (a - b) * tag))[:, None] * Y)
        nxt = np.zeros_like(v)
        for i in range(total):
            old = (s["start"] + i) % 2
            for new in [old, 1 - old]:
                idx = i + int(old != new)
                if idx <= lim:
                    op = intervals[1][new].conj().T @ intervals[0][old]
                    nxt[i] += weight @ v[idx] @ channel(op).T
        v = expm_multiply(nu * t1 * evolution(a, intervals[0]), nxt.ravel()).reshape(total, N, 4)
        src = (w * gates[0] * np.exp(-1j * a * tag)) @ Y
        return src @ v[0] @ bloch(s["initial"]).ravel()

    analyzer = I if s["analyzer"] is None else bloch(s["analyzer"])
    num = numerator(q["wave"], analyzer)
    if s["read"] == "selected":
        denominator = numerator([0, 0], I)
    else:
        denominator = (w * gates[0]) @ (
            Y @ expm_multiply(nu * t1 * generator(order, step, 0, q["hand"]), adj @ gates[1])
        )
    return float((np.exp(-1j * q["phase"]) * num / denominator).real)
