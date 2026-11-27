"""Prediction model. The rotating estimate uses the dilute-field bulk limit."""
import numpy as np
from scipy.optimize import minimize_scalar

from longitudinal import axial_signal


def rotary(q, pinning):
    h, sense = q["amplitude"], q["sense"]
    normal = q.get("normal", 0.)
    front = np.sqrt(3.) * h * (1+abs(normal)) / pinning
    winding = np.sqrt(2.) * (1. if normal else q["cutting"])
    exponent = 1 + 1j*sense*winding

    def phasors(depth):
        u = np.maximum(1-depth/front, 0.)
        b = np.zeros_like(u, dtype=complex)
        active = u > 0
        b[active] = h*np.exp(exponent*np.log(u[active]))
        return b, sense*b*u*front/(1+exponent)

    if q["readout"] == "sheet_power":
        _, e = phasors(np.asarray(q["coating"]))
        return q["conductance"] * abs(e)**2
    x = q["depth"]
    if q["readout"] == "normal_voltage":
        d = min(x, front)
        b, e = phasors((np.arange(64)+.5)*d/64)
        coupling = q["cutting"]**-2 - 1
        return d * np.mean(coupling*normal*(b.conjugate()*e).real /
                           (normal**2 + abs(b/q["cutting"])**2))
    b, e = phasors(np.asarray(x))
    value = b if q["readout"] == "induction" else e
    return value.real if q["axis"] == "y" else value.imag


class Model:
    def fit(self, records):
        def error(j):
            return sum((axial_signal(r["input"], j)-r["value"])**2 for r in records)
        self.pinning = float(minimize_scalar(error, bounds=(.85, 1.15), method="bounded").x)
        return self

    def predict(self, experiments):
        return np.array([axial_signal(q, self.pinning) if "bias" in q
                         else rotary(q, self.pinning) for q in experiments])
