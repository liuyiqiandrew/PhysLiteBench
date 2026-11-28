"""Complex envelopes, ground spinors and polarization rotations."""
import numpy as np


def pairs(values):
    a = np.asarray(values, dtype=float)
    return a[..., 0] + 1j * a[..., 1]


def spin(angles):
    t, p = angles
    return np.array([np.cos(t), np.exp(1j * p) * np.sin(t)])


def mixer(pair):
    c, s = spin(pair)
    return np.array([[c, -s.conjugate()], [s, c]])
