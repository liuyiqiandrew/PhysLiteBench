"""Direct finite signed spin and body kicks in the original playback order."""
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import expm_multiply
from reference import grid, generator

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.diag([1, -1])
I = np.eye(2)


def spin(v):
    return v[0] * sx + v[1] * sy + v[2] * sz


def ref_replay(q, nu, order=7, step=0.002):
    u, w, Y, adj, neighbors = grid(order, step)
    n = Y.shape[1]
    N = 4 * n
    t1, t2 = q["times"]
    k1, k2 = q["wave"]
    s = q["replay"]
    tag = u @ q["tag"]
    gates = [(1 + u @ g) ** 2 / 4 for g in q["gates"]]
    origin = (w * gates[0] * np.exp(-1j * k1 * tag)) @ Y
    middle = csr_matrix(adj @ ((gates[1] * np.exp(1j * (k1 - k2) * tag))[:, None] * Y))
    right = np.kron((adj @ np.exp(1j * k2 * tag))[:, None], np.eye(4))
    norm = (w * gates[0]) @ (
        Y @ expm_multiply(nu * t1 * generator(order, step, 0, q["hand"]), adj @ gates[1])
    )
    rho = (I + spin(s["initial"])) / 2
    A = I if s["analyzer"] is None else (I + spin(s["analyzer"])) / 2
    c = [1, np.exp(1j * s["control_phase"])]

    def playback(arm, U):
        # At the start of an arc the chronologically first kick acts on
        # the input index on the forward arm and on the output index on replay.
        return U.T if arm == 0 else U

    def orbital(k):
        trans = csr_matrix(adj @ ((-k * k * (u**2 @ [0.18, 0.27, 0.36]))[:, None] * Y))
        moves = []
        for dirs in neighbors:
            moves.append(
                [
                    csr_matrix(adj @ (np.exp(1j * k * q["hand"] * dist)[:, None] * basis))
                    for basis, dist in dirs
                ]
            )
        return trans, moves

    orb = [orbital(k) for k in [k1, k2]]

    def evolve(v, side, pair):
        i, j = pair
        trans, moves = orb[side]
        G = kron(trans, eye(4), format="csr")
        for d, r in enumerate([0.55, 0.82, 1.12]):
            for move, eps in zip(moves[d], [step, -step]):
                kicks = [expm(-0.5j * eps * a * spin(s["coils"][d])) for a in [1, s["sense"]]]
                small = np.kron(playback(i, kicks[i]), playback(j, kicks[j]).conj())
                G += r / step**2 * kron(move, small, format="csr")
        G -= 2 * sum([0.55, 0.82, 1.12]) / step**2 * eye(N)
        G.data[abs(G.data) < 5.0e-8] = 0
        G.eliminate_zeros()
        return expm_multiply(nu * q["times"][side] * G, v)

    pulse = [expm(-0.5j * spin(s[k])) for k in ("forward_pulse", "replay_pulse")]
    total = 0j
    for a, b in itertools.product(range(2), repeat=2):
        tail = evolve(right, 1, (a, b))
        mid = (
            kron(middle, np.kron(playback(a, pulse[a]), playback(b, pulse[b]).conj()), format="csr")
            @ tail
        )
        arr = (kron(csr_matrix(origin[None, :]), eye(4)) @ evolve(mid, 0, (a, b))).reshape(2, 2, 2, 2)
        aidx = (2, 0) if a == 0 else (0, 2)
        bidx = (3, 1) if b == 0 else (1, 3)
        arr = arr.transpose(aidx + bidx)
        total += c[a] * c[b].conjugate() * np.einsum("ca,bd,abcd->", A, rho, arr)
    return float((np.exp(-1j * q["phase"]) * total / (4 * norm)).real)
