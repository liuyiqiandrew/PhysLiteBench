"""Slab field and voltage predictions."""
import numpy as np
from scipy.optimize import minimize_scalar

from laminate import rotary
from longitudinal import axial_signal


def signal(q, pinning):
    if "bias" in q:
        return axial_signal(q, pinning)
    value = rotary(q, pinning)
    if "film" in q and q["readout"] == "sheet_power":
        # First-order film work evaluated in the normal-sheet fields.
        probe = dict(q, readout="electric", depth=q["coating"])
        field = [rotary(dict(probe, axis=axis), pinning) for axis in ("y", "z")]
        value += q["film"] * np.linalg.norm(field)
    return value


class Model:
    def fit(self, records):
        def error(j):
            return sum((signal(r["input"], j) - r["value"])**2 for r in records)
        self.pinning = float(minimize_scalar(error, bounds=(.85, 1.15), method="bounded").x)
        return self

    def predict(self, experiments):
        return np.array([signal(q, self.pinning) for q in experiments])
