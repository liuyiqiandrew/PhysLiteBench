"""Approximation that scatters the input photons in two successive isolated passages."""
from itertools import product
import numpy as np
from apparatus import fields, one_photon, pulse, signal, spin
from optics import Calibrated, calibration_signal


def sequential(q):
    incoming, output = fields(q)
    ground = spin(q["atom"])
    analyzer = spin(q["analyzer"]) if q["analyzer"] is not None else None
    kick = pulse(q)[:2, :2]
    mass = np.zeros((2, 2))
    for c, d in product(range(2), repeat=2):
        detectors = (output[0, :, c], output[1, :, d])
        amplitude = np.zeros(2, complex)
        for j in range(2):
            first = one_photon(q, incoming[0], detectors[j])
            second = one_photon(q, incoming[1], detectors[1 - j])
            amplitude += second @ kick @ first @ ground
        projection = amplitude if analyzer is None else np.vdot(analyzer, amplitude)
        mass[c, d] = np.sum(np.abs(projection) ** 2)
    return signal(q, mass)[0] if mass.sum() > 0 else 0.0


class Model(Calibrated):
    def reading(self, q):
        return sequential(q) if "rates" in q else calibration_signal(q)
