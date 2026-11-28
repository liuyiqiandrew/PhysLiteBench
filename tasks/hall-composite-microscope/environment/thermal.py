"""Adjoint thermal-source fields and relaxation of terminal capacitor voltages."""
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from film import experiments, phase_product


def port_law(materials, bias, bank, gain):
    def solve(b):
        return [
            experiments({"materials": materials}, gain, b, e.reshape(2, 2).T)
            for e in np.eye(4)
        ]

    response = np.column_stack([np.r_[j.real, j.imag] for j, *_ in solve(bias)])
    probes = solve(-bias)
    sources = []
    for material, temperatures in zip(materials, bank["phase_modes"]):
        values, modes = np.linalg.eigh(material["longitudinal"])
        sources.append((modes * (values * temperatures)) @ modes.T)

    covariance = np.zeros((4, 4))
    for i, first in enumerate(probes):
        for j, second in enumerate(probes):
            for p, strength in enumerate(sources):
                local = phase_product(first, second, p)
                covariance[i, j] += 2 * np.sum(strength * local.real)
    return response, covariance


def terminal(q, gain):
    inverse = np.linalg.inv(q["capacitance"])
    load = np.asarray(q["load"])
    evolution = []
    equilibrium = []
    for b, bath in zip(q["bias"], q["baths"]):
        G, noise = port_law(q["materials"], b, bath, gain)
        A = inverse @ (G + load)
        D = inverse @ (noise + 2 * bath["load"] * load) @ inverse
        evolution.append(A)
        equilibrium.append(solve_continuous_lyapunov(A, D))
    change = expm(-q["wait"] * evolution[1])
    start = equilibrium[1] + change @ (equilibrium[0] - equilibrium[1]) @ change.T
    return A, D, start, equilibrium[1]


def voltage(q, gain):
    A, D, start, equilibrium = terminal(q, gain)
    u, v = np.asarray(q["contacts"])
    return float(u @ start @ expm(-q["lag"] * A).T @ v)
