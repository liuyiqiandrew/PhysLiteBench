"""Square-free input and output coefficients of the continuous-field transfer operator."""
from itertools import product

import numpy as np
from scipy.linalg import expm
from apparatus import fields, pulse, signal, spin


def waveguide(q, dephase=False):
    g = np.sqrt(q["rates"])
    inp, out = fields(q)
    kick = pulse(q)
    direct = np.eye(3)
    start = np.r_[spin(q["atom"]), 0j]
    receiver = spin(q["analyzer"]) if q["analyzer"] is not None else None
    mass = np.zeros((2, 2))
    for colors in product(range(2), repeat=2):
        state = np.zeros((16, 3, 1), complex)
        state[0, :, 0] = start
        for j, dt in enumerate(q["durations"]):
            a = inp[:, j]
            b = np.array([out[k, j, color] for k, color in enumerate(colors)])
            free = np.diag([0, 0, -sum(q["rates"]) / 2 - 1j * q["detuning"][j]])
            terms = {0: free}
            for i in range(2):
                absorb = np.zeros((3, 3), complex)
                absorb[2, :2] = -a[i] * g
                emit = np.zeros((3, 3), complex)
                emit[:2, 2] = b[i] * g
                terms[1 << i] = absorb
                terms[4 << i] = emit
                for k in range(2):
                    terms[(1 << i) | (4 << k)] = (b[k] @ a[i]) * direct
            gen = np.zeros((16, 3, 16, 3), complex)
            for source in range(16):
                for added, value in terms.items():
                    if not source & added:
                        gen[source | added, :, source, :] = value
            state = (expm(dt * gen.reshape(48, 48)) @ state.reshape(48, -1)).reshape(
                16, 3, -1
            )
            if j + 1 == q["control"][0]:
                if dephase:
                    # An unobserved measurement records g0, g1 or e before the pulse.
                    state = np.concatenate(
                        [state * e[None, :, None] for e in direct], axis=2
                    )
                state = kick @ state
        finale = state[-1, :2]
        projections = finale if receiver is None else receiver.conj() @ finale
        mass[colors] = np.sum(np.abs(projections) ** 2)
    return signal(q, mass)
