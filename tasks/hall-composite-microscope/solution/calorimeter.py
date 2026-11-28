"""Tilt the capacitor dynamics by the heat flowing into its load bath."""
import numpy as np
from scipy.linalg import expm
from thermal import terminal


def characteristic(
    A, D, start, inverse, load, temperature, duration, tilt, common=True
):
    n = len(A)
    # Ito heat has the constant drift -T tr(C^-1 L). The same load noise
    # drives the terminal voltage and the measured heat current.
    cross = 2j * tilt * temperature * inverse @ load
    F = A + cross if common else A
    B = -2 * (1j * tilt - tilt**2 * temperature) * load
    system = np.block([[F, D], [B, -F.T]])
    steps = max(
        1, int(np.ceil(duration * (np.linalg.norm(F, 1) + np.linalg.norm(B, 1))))
    )
    dt = duration / steps
    transfer = expm(system * dt)
    H = np.zeros_like(A, dtype=complex)
    logdet = 0j
    for _ in range(steps):
        X = transfer[:n, :n] + transfer[:n, n:] @ H
        Y = transfer[n:, :n] + transfer[n:, n:] @ H
        logdet += np.log(np.linalg.eigvals(X)).sum()
        H = np.linalg.solve(X.T, Y.T).T
    logdet += np.log(np.linalg.eigvals(np.eye(n) + start @ H)).sum()
    shift = (
        0.5 * np.trace(F) - 1j * tilt * temperature * np.trace(inverse @ load)
    ) * duration
    return np.exp(shift - 0.5 * logdet)


def heat(q, gain, common=True):
    A, D, start, _ = terminal(q, gain)
    z = characteristic(
        A,
        D,
        start,
        np.linalg.inv(q["capacitance"]),
        np.asarray(q["load"]),
        q["baths"][1]["load"],
        q["duration"],
        q["tilt"],
        common,
    )
    return float((np.exp(-1j * q["angle"]) * z).real)
