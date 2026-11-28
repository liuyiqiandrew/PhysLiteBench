"""Treat photon labels as distinguishable and use the intensity marginals of each emission."""
import numpy as np
from scipy.special import roots_hermitenorm
from gaussian import mesh
from pump import PumpSurvey


def spectral_reading(q):
    spec = q["spectral"]
    states = spec["preparations"]
    n = len(states[0]["mean"])
    which, delay, chirp = spec["plate"]
    bank = np.isin(np.arange(n), q["bank"])
    B = mesh(n, q["before"])
    A = mesh(n, q["after"])
    gate = q["gate"]
    branches = [(1.0, np.ones(n))]
    z, w = roots_hermitenorm(36)
    w = w / np.sqrt(2 * np.pi)
    if gate:
        sign = np.ones(n)
        sign[gate["rails"]] = -1
        branches = [(0.5, np.ones(n)), (0.5 * gate["parity"], sign)]

    def chi(angle):
        total = 0j
        count = np.exp(1j * angle * bank)
        for j, st in enumerate(states):
            intensity = 1.0 if j == 0 else abs(complex(*spec["weight"])) ** 2
            contributions = (
                np.zeros((len(branches), len(branches)), complex) + intensity
            )
            for i in range(n):
                x = st["mean"][i] + np.sqrt(st["cov"][i][i]) * z
                plate = np.ones((len(x), n), complex)
                plate[:, which] = np.exp(1j * (-delay * x + chirp * x**2 / 2))
                amps = [(A @ ((plate * sign * B[:, i]).T)).T for _, sign in branches]
                for l, left in enumerate(amps):
                    for k, right in enumerate(amps):
                        contributions[l, k] *= w @ np.sum(
                            left.conj() * count * right, axis=1
                        )
            for l, (a, _) in enumerate(branches):
                for k, (b, _) in enumerate(branches):
                    total += a * b * contributions[l, k]
        return total

    theta = q["lambda"]
    return float(
        (chi(theta) * np.exp(-1j * (q["phase"] + theta * sum(bank)))).real
        / chi(0.0).real
    )


class Model(PumpSurvey):
    def reading(self, q):
        return spectral_reading(q) if "spectral" in q else super().reading(q)
