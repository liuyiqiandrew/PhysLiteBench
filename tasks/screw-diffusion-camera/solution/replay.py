"""Average the two spin histories and the tagged-particle fringe on one path."""
import numpy as np
from scipy.linalg import expm
from scipy.sparse import eye, kron
from scipy.sparse.linalg import expm_multiply

from apparatus import ROT, SCREWS, TRANS
from passive import operators, project


PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
IDENTITY = np.eye(2)


def sigma(vector):
    return np.einsum("i,ijk->jk", vector, PAULI)


def replay(q, mobility, order=6):
    gyro = q["replay"]
    forward = np.array([-0.5j * sigma(v) for v in gyro["coils"]])
    reverse_adjoint = -gyro["sense"] * forward
    pulse, replay_pulse = [expm(-0.5j * sigma(gyro[k])) for k in ("forward_pulse", "replay_pulse")]
    adjoint_pulse = replay_pulse.conj().T
    rho = (IDENTITY + sigma(gyro["initial"])) / 2
    analyzer = IDENTITY if gyro["analyzer"] is None else (IDENTITY + sigma(gyro["analyzer"])) / 2

    position, rotation = operators(order)
    size = (order + 1) ** 2
    space, pair = eye(size, format="csr"), eye(4, format="csr")
    gates = []
    for axis in q["gates"]:
        weight = eye(position[0].shape[0]) + project(position, axis)
        gates.append((weight @ weight)[:size, :size] / 4)

    def spatial(k):
        solvent = -k * k * sum(d * (p @ p) for d, p in zip(TRANS, position))
        twists = []
        for j, r in enumerate(ROT):
            lever = q["hand"] * SCREWS[j] + np.cross(np.eye(3)[j], q["tag"])
            twist = rotation[j] + 1j * k * project(position, lever)
            solvent += r * (twist @ twist)
            twists.append(twist[:size, :size])
        return solvent[:size, :size], twists

    first, second = [spatial(k) for k in q["wave"]]
    t1, t2 = q["times"]
    zero, _ = spatial(0.0)
    constant = np.zeros(size)
    constant[0] = 1.0
    accepted = (gates[0] @ expm_multiply(t1 * mobility * zero, gates[1] @ constant))[0].real

    def history(left, right, pulse_left, pulse_right):
        # Transposition turns chronological spin updates into a backward product.
        spins = [(np.kron(a, IDENTITY) + np.kron(IDENTITY, b)).T for a, b in zip(left, right)]

        def evolve(v, spatial_terms, duration):
            solvent, twists = spatial_terms
            generator = kron(solvent, pair, format="csr")
            for r, twist, spin in zip(ROT, twists, spins):
                generator += r * (
                    kron(space, spin @ spin, format="csr") + 2 * kron(twist, spin, format="csr")
                )
            return expm_multiply(duration * mobility * generator, v)

        final = np.zeros((4 * size, 4), complex)
        final[:4] = np.eye(4)
        back = evolve(final, second, t2)
        middle = kron(gates[1], np.kron(pulse_left, pulse_right).T, format="csr") @ back
        back = evolve(middle, first, t1)
        return (kron(gates[0].getrow(0), pair) @ back).T.reshape(2, 2, 2, 2)

    # V=R† is chronological. Its density contribution exchanges input and output.
    ff = history(forward, forward.conj(), pulse, pulse.conj())
    vv = history(reverse_adjoint, reverse_adjoint.conj(), adjoint_pulse, adjoint_pulse.conj())
    fv = history(forward, reverse_adjoint, pulse, adjoint_pulse)
    fv_star = history(forward.conj(), reverse_adjoint.conj(), pulse.conj(), adjoint_pulse.conj())
    control = np.exp(1j * gyro["control_phase"])
    signal = np.einsum("ab,cd,bacd->", analyzer, rho, ff)
    signal += np.einsum("ab,cd,bacd->", rho, analyzer, vv)
    signal += control.conjugate() * np.einsum("ab,cd,bdca->", analyzer, rho, fv)
    signal += control * np.einsum("ab,cd,bdca->", analyzer.conj(), rho.conj(), fv_star)
    return float((np.exp(-1j * q["phase"]) * signal / (4 * accepted)).real)
