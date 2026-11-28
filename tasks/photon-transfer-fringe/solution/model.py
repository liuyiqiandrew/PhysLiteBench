"""Passive-optical overlaps of a thermofield purification."""
import numpy as np

from optics import Calibrated, mesh


def source_pairs(q):
    n, r, phase = np.array(q['source']).T
    size = len(n)
    thermal = np.sqrt(n / (1 + n))
    squeezed = -np.exp(1j*phase)*np.tanh(r)
    cross = thermal / np.cosh(r)
    pairs = np.block([[np.diag(squeezed), np.diag(cross)],
                      [np.diag(cross), np.diag(-squeezed.conj()*thermal**2)]])
    transform = np.eye(2*size, dtype=complex)
    transform[:size, :size] = mesh(size, q['prep'])
    return transform @ pairs @ transform.T


class Model(Calibrated):
    def reading(self, q):
        pairs = source_pairs(q)
        size = len(q['source'])
        mean, squeeze, _ = np.array(q['source']).T
        norm = np.prod(1/((1+mean)*np.cosh(squeeze)))
        before = mesh(size, q['before'])
        after = mesh(size, q['after'])
        bank = np.zeros(size)
        bank[q['bank']] = 1
        phase = np.arange(96)*2*np.pi/96
        rotation = np.exp(1j*phase[:, None]*np.r_[bank, np.zeros(size)])
        states = rotation[:, :, None]*pairs*rotation[:, None, :]

        def overlap(operator):
            extension = np.eye(2*size, dtype=complex)
            extension[:size, :size] = operator
            propagated = extension @ states @ extension.T
            eigenvalues = np.linalg.eigvals(states.conj().transpose(0, 2, 1) @ propagated)
            return norm * np.mean(np.exp(-.5*np.log1p(-eigenvalues).sum(axis=-1)))

        d = np.exp(1j*q['lambda']*bank)
        final = after.conj().T @ (d[:, None]*after)
        right = before*d.conj()
        gate = q['gate']
        if gate is None:
            value = overlap(before.conj().T @ final @ right)
        else:
            sign = gate['parity']
            parity = np.ones(size)
            parity[gate['rails']] = -1
            arms = [before, parity[:, None]*before]
            value = sum(sign**(i+j)*overlap(left.conj().T @ final @ (arm*d.conj()))
                        for i, left in enumerate(arms) for j, arm in enumerate(arms))
            retention = (1 + sign*overlap(before.conj().T @ arms[1]).real)/2
            value /= 4*retention
        return float((np.exp(-1j*q['phase'])*value).real)
