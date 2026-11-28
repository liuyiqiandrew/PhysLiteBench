"""Passive mesh convention and calibration on a thermal input mixed with vacuum."""
import numpy as np


def mesh(size, elements):
    matrix = np.eye(size, dtype=complex)
    for i, j, angle, phase in elements:
        c = np.cos(angle)
        s = np.sin(angle) * np.exp(1j * phase)
        matrix[[i, j]] = np.array([[c, -s.conjugate()], [s, c]]) @ matrix[[i, j]]
    return matrix


def calibration_signal(q):
    n = q['source'][0][0]
    p = np.sin(q['before'][0][2])**2
    characteristic = 1 / (1 + n * p * (1 - np.exp(-1j * q['lambda'])))
    return (np.exp(-1j * q['phase']) * characteristic).real


class Calibrated:
    def fit(self, records):
        signals = np.array([calibration_signal(row['input']) for row in records])
        values = np.array([row['value'] for row in records])
        self.visibility = float(signals @ values / (signals @ signals))
        return self

    def predict(self, inputs):
        return self.visibility * np.array([self.reading(q) for q in inputs], dtype=float)
