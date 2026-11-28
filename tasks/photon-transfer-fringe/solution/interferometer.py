"""Fold four encounters with one atom onto a shared position along the optical delay."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply
from apparatus import mixer, pairs, spin


def lifted(a, site):
    return kron(eye(3**site), kron(csr_matrix(a), eye(3 ** (3 - site))), format="csr")


def router(q):
    time = pairs(q["temporal"])
    r = np.zeros((4, 4), complex)
    for j, (angle, alpha, phase) in enumerate(q["arms"]):
        pol = np.exp(1j * phase) * mixer((angle, alpha))
        for k in range(2):
            r[2 * k : 2 * k + 2, 2 * j : 2 * j + 2] = time[k, j] * pol
    return r


def delay_signal(q, measure=False):
    rates = q["rates"]
    S = router(q)
    angles = q["control"]
    gap, delta = q["gap"]
    n = len(q["detuning"][0])
    dt = q["duration"] / n
    reverse = q["input_reverse"] + q["output_reverse"]
    init = np.r_[spin(q["atom"]), 0j]
    kick = np.zeros((3, 3), complex)
    kick[:2, :2] = mixer(angles[:2])
    kick[2, 2] = np.exp(1j * angles[2] - (sum(rates) / 2 + 1j * delta) * gap)
    raw_in = pairs(q["input_modes"]) / np.sqrt(dt)
    raw_out = pairs(q["output_mode"]).conj() / np.sqrt(dt)
    pol_in = mixer(q["input_rotation"])
    pol_out = mixer(q["output_rotation"])
    energies = q["detuning"] + q["return_detuning"]
    id81 = eye(81, format="csr")
    # Optical operators on each replica, with reversed chronological indices for each reversed arm.
    lower = []
    raise_ = []
    exc = []
    for l, rev in enumerate(reverse):
        e = np.diag([0, 0, 1])
        exc.append(lifted(e, l))
        for c in range(2):
            z = np.zeros((3, 3))
            z[c, 2] = np.sqrt(rates[c])
            lower.append(lifted(z.T if rev else z, l))
            raise_.append(lifted(z if rev else z.T, l))
    absorb = [
        raise_[c] + sum(S[d, c] * raise_[4 + d] for d in range(4)) for c in range(4)
    ]
    emit = [lower[4 + d] + sum(S[d, c] * lower[c] for c in range(4)) for d in range(4)]
    capture = sum(
        (-S[d, c]) * (lower[c] @ raise_[4 + d]) for d in range(4) for c in range(4)
    )
    # Raising operators for four distinct formal field labels. Coefficients containing a repeated label are discarded.
    add = {}
    for bits in (0, 1, 2, 4, 8, 5, 9, 6, 10):
        mask = np.zeros((16, 16))
        for src in range(16):
            if not (bits & src):
                mask[src | bits, src] = 1
        add[bits] = csr_matrix(mask)
    pulses = {}
    for j in range(n):
        idx = [n - j - 1 if r else j for r in reverse]
        A = np.stack(
            [
                np.concatenate([pol_in[:, i] * raw_in[i, b, idx[b]] for b in range(2)])
                for i in range(2)
            ]
        )
        H = np.stack(
            [
                np.concatenate(
                    [pol_out[c, :] * raw_out[b, idx[2 + b]] for b in range(2)]
                )
                for c in range(2)
            ]
        )
        K = capture + sum(
            (-sum(rates) / 2 - 1j * energies[l][idx[l]]) * exc[l] for l in range(4)
        )
        base = kron(add[0], K, format="csr")
        for i in range(2):
            base += kron(
                add[1 << i], -sum(z * g for z, g in zip(A[i], absorb)), format="csr"
            )
        pulses[j] = (A, H, base)
    M = []
    for colors in combinations_with_replacement(range(2), 2):
        state = np.zeros((1296, 81), complex)
        state[:81] = np.eye(81)
        for j, (A, H, base) in pulses.items():
            K = base.copy()
            for k, c in enumerate(colors):
                K += kron(
                    add[4 << k], sum(z * g for z, g in zip(H[c], emit)), format="csr"
                )
                for i in range(2):
                    K += kron(
                        add[(4 << k) | (1 << i)], (H[c] @ S @ A[i]) * id81, format="csr"
                    )
            state = expm_multiply(dt * K, state, traceA=dt * K.diagonal().sum())
        axes = list(range(8))
        for l, rev in enumerate(reverse):
            if rev:
                axes[l], axes[l + 4] = axes[l + 4], axes[l]
        T = state[-81:].reshape([3] * 8).transpose(axes)
        # Unfold in the physical order 0,1,control,2,3.
        atomic = []
        sectors = [np.diag([1, 1, 0]), np.diag([0, 0, 1])] if measure else [np.eye(3)]
        for sector in sectors:
            amp = np.einsum(
                "abcdeafc,fb,e->d", T, kick @ sector, init, optimize="greedy"
            )[:2]
            if colors[0] == colors[1]:
                amp /= np.sqrt(2)
            atomic.append(
                amp if q["analyzer"] is None else np.vdot(spin(q["analyzer"]), amp)
            )
        M.append(np.sum(np.abs(atomic) ** 2))
    p = sum(M)
    y = np.dot(M, np.cos(q["lambda"] * np.array([1, 0, -1]) - q["phase"]))
    return float(y / p if q["conditional"] else y), float(p)
