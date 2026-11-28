"""Continuous-time scattering paths with zero, one, or two excited-state dwell intervals."""
from itertools import permutations, product
import numpy as np
from scipy.linalg import expm


def rotation(angles):
    a, p = angles
    return np.array(
        [
            [np.cos(a), -np.sin(a) * np.exp(-1j * p)],
            [np.sin(a) * np.exp(1j * p), np.cos(a)],
        ]
    )


def envelopes(rows, durations):
    return np.array([[complex(*z) for z in row] for row in rows]) / np.sqrt(durations)


def reading(q):
    intervals = q["durations"]
    source = envelopes(q["input_modes"], intervals)
    filters = envelopes(q["output_modes"], intervals)
    polarized = rotation(q["input_rotation"])
    analyzer = [rotation(x) for x in q["output_rotations"]]
    init = np.r_[rotation(q["atom"])[:, 0], 0j]
    pulse_at, theta, phase, stark = q["control"]
    kick = np.zeros((3, 3), complex)
    kick[:2, :2] = rotation((theta, phase))
    kick[2, 2] = np.exp(1j * stark)
    rates = np.asarray(q["rates"])
    absorption = np.zeros((2, len(intervals), 3, 3), complex)
    absorption[:, :, 2, :2] = (
        -source[:, :, None] * (polarized.T * np.sqrt(rates))[:, None]
    )
    total = moment = 0.0
    for c0, c1 in product(range(2), repeat=2):
        polar_bras = np.array([analyzer[0][c0], analyzer[1][c1]])
        detected = np.conjugate(filters)[:, :, None] * polar_bras[:, None, :]
        emission = np.zeros_like(absorption)
        emission[:, :, :2, 2] = detected * np.sqrt(rates)
        overlap = (filters.conj() @ (source * intervals).T) * (polar_bras @ polarized)

        def ordered(word):
            count = len(word) + 1
            state = np.zeros((count, 3), complex)
            state[0] = init
            for t, (width, delta) in enumerate(zip(intervals, q["detuning"])):
                free = np.diag([0, 0, -rates.sum() / 2 - 1j * delta])
                evolution = np.kron(np.eye(count), free)
                for n, event in enumerate(word):
                    evolution[3 * n + 3 : 3 * n + 6, 3 * n : 3 * n + 3] = event[t]
                state = (expm(width * evolution) @ state.ravel()).reshape(count, 3)
                if t + 1 == pulse_at:
                    state = (kick @ state.T).T
            return state[-1]

        spin_out = (overlap[0, 0] * overlap[1, 1] + overlap[0, 1] * overlap[1, 0]) * (
            kick @ init
        )
        for i, j in product(range(2), repeat=2):
            spin_out += overlap[1 - j, 1 - i] * ordered([absorption[i], emission[j]])
        for i, k in permutations(range(2)):
            for j, l in permutations(range(2)):
                spin_out += ordered(
                    [absorption[i], emission[j], absorption[k], emission[l]]
                )
        v = spin_out[:2]
        if q["analyzer"] is not None:
            v = np.vdot(rotation(q["analyzer"])[:, 0], v)
        prob = np.square(np.abs(v)).sum()
        total += prob
        moment += prob * np.cos(q["lambda"] * (1 - c0 - c1) - q["phase"])
    return float(moment / total), float(total)
