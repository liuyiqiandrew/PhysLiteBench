"""Continuous-mode Fock scattering on one pass through a Λ atom."""
from itertools import product
import numpy as np
from scipy.linalg import expm


def transfer(rates, durations, detunings, incoming, outgoing):
    """Atomic transfer matrix for labelled ket envelopes and outgoing bra envelopes.

    Each entry of incoming or outgoing is (time interval, polarization). Fields
    are continuous amplitudes on each interval. Outgoing entries already include
    complex conjugation of the temporal receiver. Repeating an identical mode
    applies its creation or annihilation operator repeatedly, without a factorial.
    """
    coupling = np.sqrt(rates)
    count = 1 << (len(incoming) + len(outgoing))
    state = np.zeros((count, 3, 3), complex)
    state[0] = np.eye(3)
    for j, dt in enumerate(durations):
        terms = {0: np.diag([0, 0, -sum(rates) / 2 - 1j * detunings[j]])}
        for a, f in enumerate(incoming):
            m = np.zeros((3, 3), complex)
            m[2, :2] = -coupling * f[j]
            terms[1 << a] = m
        for b, h in enumerate(outgoing):
            bit = 1 << (len(incoming) + b)
            m = np.zeros((3, 3), complex)
            m[:2, 2] = coupling * h[j]
            terms[bit] = m
            for a, f in enumerate(incoming):
                terms[bit | (1 << a)] = np.eye(3) * (h[j] @ f[j])
        gen = np.zeros((count, 3, count, 3), complex)
        for mask in range(count):
            for added, value in terms.items():
                if not mask & added:
                    gen[mask | added, :, mask, :] = value
        state = (
            expm(dt * gen.reshape(3 * count, 3 * count)) @ state.reshape(3 * count, 3)
        ).reshape(count, 3, 3)
    return state[-1]
