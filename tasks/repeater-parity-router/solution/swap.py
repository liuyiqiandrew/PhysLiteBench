"""Outcome-resolved Choi amplitudes and the erasure of a common source marker."""
import numpy as np
import math
from apparatus import arm, pair, PAULI


def weights(q, a, shortcut=None):
    bank = [pair(v) for v in q["bank"]]
    overlap = np.vdot(bank[1], bank[0])
    par = q["parity"]
    lam = a * q["loading"]
    prob = np.exp(-lam)
    ready = 0.0
    coh = 0j
    for n in range(1, 41):
        prob *= lam / n
        if n >= 2 and (par is None or n % 2 == par):
            ready += prob
            coh += prob * overlap ** (n - 2)
    z = q["marker"]["sign"] * np.exp(1j * q["marker"]["phase"]) * coh / ready
    if shortcut == "mixture":
        z = 0j
    if shortcut == "erase_all":
        z = q["marker"]["sign"] * np.exp(1j * q["marker"]["phase"])
    return bank, np.array([[1.0, z], [z.conjugate(), 1.0]]) / 4


def values(q, a, shortcut=None):
    bank, coef = weights(q, a, shortcut)
    g = np.vdot(bank[1], bank[0])
    marker = float(np.real(0.5 + 2 * coef[0, 1] * g * g))
    matrices = []
    for channel in (0, 1):
        out = []
        for b in (0, 1):
            out.append(
                [
                    arm(
                        q["outer"][channel],
                        m,
                        q["analyzers"][channel],
                        b,
                        reverse=bool(channel),
                        correction=shortcut != "no_feedforward",
                    )
                    for m in bank
                ]
            )
        matrices.append(out)
    probes = [np.kron(*[PAULI[s] for s in word]) for word in q["spin"]]
    jointop = np.kron(*probes)
    single0 = np.kron(probes[0], np.eye(4))
    single1 = np.kron(np.eye(4), probes[1])
    prob = moment = left_mom = right_mom = 0.0
    for b in (0, 1):
        for d in (0, 1):
            if q["report"] is not None and b ^ d != q["report"]:
                continue
            psi = np.array(
                [np.kron(matrices[0][b][j], matrices[1][d][j]) for j in (0, 1)]
            )
            rho = psi.T @ coef @ psi.conj()
            prob += np.trace(rho).real
            moment += np.trace(jointop @ rho).real
            left_mom += np.trace(single0 @ rho).real
            right_mom += np.trace(single1 @ rho).real
    return dict(
        marker=marker,
        herald=prob / marker,
        fringe=(
            left_mom * right_mom / prob**2
            if shortcut == "independent_outputs"
            else moment / prob
        ),
    )
