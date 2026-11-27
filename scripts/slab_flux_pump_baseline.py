"""Superpose the static self-field and the zero-transport cyclic response."""
import numpy as np
from scipy.optimize import minimize_scalar


def transform(b):
    return b * (1 + abs(b) / 2)


def induction(u):
    return 2 * u / (1 + np.sqrt(1 + 2 * abs(u)))


def signal(q, pinning):
    if q["readout"] == "impulse":
        # The bias is static. Faraday's law and symmetry give zero cycle
        # impulse on the unbiased branch, which returns to the same profile.
        return 0.0
    x = -1 + (np.arange(2048) + 0.5) / 1024
    distance = pinning * (1 - abs(x))
    transport = -np.sign(x * q["bias"]) * induction(np.maximum(
        0, abs(transform(q["bias"])) - distance))
    applied = np.zeros_like(x)
    turns = [q["low"], q["high"], q["low"], q["high"]]
    if q["readout"] == "trough":
        turns.append(q["low"])
    for h in turns:
        applied = np.clip(applied, transform(h) - distance, transform(h) + distance)
    return np.mean(induction(applied) + transport)


class Model:
    def fit(self, records):
        def error(pinning):
            return sum((signal(r["input"], pinning) - r["value"])**2 for r in records)
        self.pinning = float(minimize_scalar(error, bounds=(0.85, 1.15), method="bounded").x)
        return self

    def predict(self, experiments):
        return np.array([signal(q, self.pinning) for q in experiments])
