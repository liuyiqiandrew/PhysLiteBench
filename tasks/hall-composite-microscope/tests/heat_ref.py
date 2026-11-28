"""Gaussian integration of the finite midpoint heat meter, followed by its continuum limit."""
import numpy as np
from laminate import noisy_state


def midpoint(C, G, N, L, T, start, duration, k, steps):
    n = len(C)
    dt = duration / steps
    bath = np.linalg.cholesky(2 * T * L)
    B = np.column_stack((np.linalg.cholesky((N + N.T) / 2), -bath))
    update = np.linalg.solve(
        C + 0.5 * dt * (G + L),
        np.column_stack((C - 0.5 * dt * (G + L), np.sqrt(dt) * B)),
    )
    initial = np.column_stack((np.eye(n), np.zeros((n, 2 * n))))
    mid = 0.5 * (update + initial)
    current = np.column_stack((np.zeros((n, 2 * n)), bath))
    heat = dt * mid.T @ L @ mid + 0.5 * np.sqrt(dt) * (
        mid.T @ current + current.T @ mid
    )
    K = np.zeros((n, n))
    amp = 0j
    for _ in range(steps):
        H = update.T @ K @ update - 2j * k * heat
        integrate = np.eye(2 * n) + H[n:, n:]
        amp -= 0.5 * np.log(np.linalg.eigvals(integrate)).sum()
        K = H[:n, :n] - H[:n, n:] @ np.linalg.solve(integrate, H[n:, :n])
    return np.exp(amp - 0.5 * np.log(np.linalg.eigvals(np.eye(n) + start @ K)).sum())


def heat_probe(q, gain, steps=48):
    G, N, _, start = noisy_state(q, gain)
    C = np.asarray(q["capacitance"])
    L = np.asarray(q["load"])
    T = q["baths"][1]["load"]
    # The time integral of the sampled heat has a regular weak error in dt.
    # Extrapolate the independently integrated finite meters in successive halvings.
    vals = [
        midpoint(C, G, N, L, T, start, q["duration"], q["tilt"], steps * 2**i)
        for i in range(4)
    ]
    for order in range(1, 4):
        vals = [b + (b - a) / (2**order - 1) for a, b in zip(vals, vals[1:])]
    return float((np.exp(-1j * q["angle"]) * vals[0]).real)
