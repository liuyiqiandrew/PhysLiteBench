"""Propagate the reduced source qubit through unobserved emissions and project the complete optical arms."""
import numpy as np, math


def ry(a):
    h = a / 2
    return np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])


def state(values):
    b, t, p = values
    return np.kron(np.array([1, 0]), ry(t)[:, 0]) * math.cos(b) + np.exp(
        1j * p
    ) * np.kron(np.array([0, 1]), ry(t)[:, 1]) * math.sin(b)


def gate(tensor, axis, M):
    return np.moveaxis(np.tensordot(M, tensor, axes=([1], [axis])), 0, axis)


def reference(q, a):
    b0, b1 = [state(v) for v in q["bank"]]
    A, C = [state(v) for v in q["outer"]]
    # First trace N-2 discarded copies in the computational basis by their four Kraus operators.
    dist = np.exp(-a * q["loading"]) * (a * q["loading"]) ** 2 / 2
    env = np.array([[0.5, 0.5], [0.5, 0.5]], complex)
    mixed = np.zeros_like(env)
    norm = 0.0
    for n in range(2, 36):
        if q["parity"] is None or n % 2 == q["parity"]:
            mixed += dist * env
            norm += dist
        nextenv = np.zeros_like(env)
        for v in range(4):
            K = np.diag([b0[v], b1[v]])
            nextenv += K @ env @ K.conj().T
        env = nextenv
        dist *= a * q["loading"] / (n + 1)
    h = np.array(
        [1, q["marker"]["sign"] * np.exp(1j * q["marker"]["phase"])]
    ) / np.sqrt(2)
    marker_op = np.diag(h.conj())
    marked = marker_op @ mixed @ marker_op.conj().T / norm
    vals, vecs = np.linalg.eigh(marked)
    # Build the pair states in the physical bank port order L,R, including arm 1 wired from R to L.
    branches = [
        np.kron(np.kron(A, bank), np.kron(C, bank)).reshape((2,) * 8).swapaxes(6, 7)
        for bank in (b0, b1)
    ]
    X = np.array([[0, 1], [1, 0]], complex)
    ops = {
        "X": X,
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
        "I": np.eye(2),
    }
    measure = [ops[x] for pair in q["spin"] for x in pair]
    marker = accepted = signal = 0.0
    for k, p in enumerate(vals):
        if p <= 1e-16:
            continue
        psi = sum(vecs[j, k] * branches[j] for j in (0, 1))
        marker += p * np.vdot(psi, psi).real
        for axis, angle in ((2, q["analyzers"][0]), (6, q["analyzers"][1])):
            psi = gate(psi, axis, ry(angle))
        for b in (0, 1):
            for d in (0, 1):
                if q["report"] is not None and (b + d) % 2 != q["report"]:
                    continue
                # First and second conclusive measurements are actual Bell bras; erase their labels only after the two corrections.
                first = (psi[:, 0, 1] + (-1) ** b * psi[:, 1, 0]) / np.sqrt(2)
                psi4 = (
                    first[:, :, :, 0, 1, :] + (-1) ** d * first[:, :, :, 1, 0, :]
                ) / np.sqrt(2)
                for axis, bit in ((1, b), (3, d)):
                    if bit:
                        psi4 = gate(psi4, axis, ops["Z"])
                    psi4 = gate(psi4, axis, X)
                accepted += p * np.vdot(psi4, psi4).real
                observed = psi4
                for axis, pauli in enumerate(measure):
                    observed = gate(observed, axis, pauli)
                signal += p * np.vdot(psi4, observed).real
    return dict(marker=marker, herald=accepted / marker, fringe=signal / accepted)
