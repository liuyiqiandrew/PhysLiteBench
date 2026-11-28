"""Spin-3/2 matrices and the two bases of the physical analyzer."""
import numpy as np
from scipy.linalg import expm

raise_m = np.diag([np.sqrt(3), 2.0, np.sqrt(3)], 1)
spin_axes = np.array(
    [0.5 * (raise_m + raise_m.T), -0.5j * (raise_m - raise_m.T), np.diag([1.5, 0.5, -0.5, -1.5])]
)


def bands(setting):
    rotation = expm(-1j * sum(g * a for g, a in zip(setting["turn"], spin_axes)))
    selected = [1, 2] if setting["band"] == "central" else [0, 1]
    rest = [j for j in range(4) if j not in selected]
    return rotation[:, selected], rotation[:, rest]
