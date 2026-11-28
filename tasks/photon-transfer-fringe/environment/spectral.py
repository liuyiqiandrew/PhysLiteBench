"""Frequency-resolved exchange paths in an achromatic optical network."""
from itertools import permutations, product
import numpy as np
from optics import mesh


def forms(spec):
    states = []
    for prep in spec["preparations"]:
        cov = np.array(prep["cov"])
        mu = np.array(prep["mean"])
        precision = 0.5 * np.linalg.inv(cov)
        states.append(
            (
                precision - 1j * np.array(prep["chirp"]),
                precision @ mu - 1j * np.array(prep["delay"]),
                -0.5 * mu @ precision @ mu - 0.25 * np.linalg.slogdet(cov)[1],
            )
        )
    if len(states) == 1:
        return states, np.ones((1, 1))
    amplitude = complex(*spec["weight"])
    marker = complex(*spec["marker"])
    weight = np.array(
        [
            [1, amplitude * marker],
            [(amplitude * marker).conjugate(), abs(amplitude) ** 2],
        ]
    )
    return states, weight


def overlaps(q, states, weights, perm, shift):
    n = len(perm)
    delay, chirp = q["spectral"]["plate"][1:]
    zz = np.zeros(len(shift), complex)
    for a, (Pa, la, ca) in enumerate(states):
        for b, (Pb, lb, cb) in enumerate(states):
            if abs(weights[a, b]) == 0:
                continue
            base = Pa.conj() + Pb[np.ix_(perm, perm)]
            linear = la.conj() + lb[perm] - 1j * delay * shift
            matrix = np.broadcast_to(base, (len(shift), n, n)).copy()
            di = np.arange(n)
            matrix[:, di, di] -= 1j * chirp * shift
            # Continuous square-root branch from zero chirp.
            L = np.linalg.cholesky(base.real)
            inv = np.linalg.inv(L)
            eig = np.linalg.eigvalsh(inv @ matrix.imag @ inv.T)
            detroot = np.prod(np.diag(L)) * np.prod(np.sqrt(1 + 1j * eig), axis=1)
            x = np.linalg.solve(matrix, linear[..., None])[..., 0]
            zz += (
                weights[a, b]
                * np.exp(ca + cb + 0.5 * np.sum(linear * x, axis=1))
                / detroot
            )
    return zz


def spectral_signal(q):
    spec = q["spectral"]
    states, weights = forms(spec)
    n = len(states[0][1])
    bank = np.isin(np.arange(n), q["bank"])
    entrance = int(sum(bank))
    which = spec["plate"][0]
    before = mesh(n, q["before"])
    after = mesh(n, q["after"])
    gate = q["gate"]
    diag = np.ones(n)
    terms = [(1.0, np.ones(n))]
    if gate is not None:
        diag[gate["rails"]] = -1.0
        terms = [(0.5, np.ones(n)), (0.5 * gate["parity"], diag)]
    branches = []
    for coefficient, projection in terms:
        basis = after * projection
        covered = np.outer(basis[:, which], before[which, :])
        branches.append((coefficient, [basis @ before - covered, covered]))
    perms = [np.array(x) for x in permutations(range(n))]
    shift = np.array(list(product([-1, 0, 1], repeat=n)))
    idx = np.arange(n)
    integrals = [overlaps(q, states, weights, p, shift) for p in perms]

    def characteristic(angle):
        out = np.zeros((n, n, 3), complex)
        total = 0j
        count = np.exp(1j * angle * bank)
        for wa, a in branches:
            for wb, b in branches:
                out.fill(0.0)
                for h in (0, 1):
                    for k in (0, 1):
                        out[:, :, k - h + 1] += (a[h].conj().T * count) @ b[k]
                for p, z in zip(perms, integrals):
                    total += (
                        wa * wb * np.sum(z * np.prod(out[idx, p, shift + 1], axis=1))
                    )
        return total

    mass = characteristic(0.0).real
    norm = integrals[0][len(shift) // 2].real
    theta = q["lambda"]
    signal = (
        np.exp(-1j * (q["phase"] + theta * entrance)) * characteristic(theta)
    ).real / mass
    return float(signal), float(mass / norm)
