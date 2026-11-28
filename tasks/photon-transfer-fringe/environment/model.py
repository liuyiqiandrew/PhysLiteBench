"""Linear spin-wave storage, with the waveguide atom projected onto its ground doublet at handoff."""
from itertools import combinations_with_replacement, permutations
import numpy as np
from apparatus import mixer, pairs, spin
from optics import Calibrated, calibration_signal
from single_pass import transfer


def linear_return(q):
    first, second, store = q["first"], q["second"], q["store"]
    dt, dt2 = np.sqrt(first["durations"]), np.sqrt(second["durations"])
    incoming = [
        f[:, None] * p
        for f, p in zip(
            pairs(first["input_modes"]) / dt, mixer(first["input_rotation"]).T
        )
    ]
    write = (pairs(first["write_mode"]).conj() / dt)[:, None] * np.eye(2)[:, None, :]
    read = (pairs(second["read_mode"]) / dt2)[:, None] * np.eye(2)[:, None, :]
    out = (pairs(second["output_mode"]).conj() / dt2)[:, None] * mixer(
        second["output_rotation"]
    )[:, None, :]
    m = store["cells"]
    pos = np.arange(m) - (m - 1) / 2
    k = mixer(store["rotation"])
    for b in range(2):
        for a in range(2):
            form = np.mean(
                np.exp(1j * pos * (store["write_k"][a] - store["read_k"][b]))
            )
            k[b, a] *= (
                -np.sin(np.sqrt(m) * store["write"][a])
                * np.sin(np.sqrt(m) * store["read"][b])
                * np.exp(1j * store["phase"])
                * form
            )
    initial = np.r_[spin(q["atom"]), 0j]
    rot = mixer(q["control"][:2])
    basis = list(combinations_with_replacement(range(2), 2))
    replay = []

    def norm(i):
        return np.sqrt(2.0) if i[0] == i[1] else 1.0

    for a in basis:
        first_op = transfer(
            q["rates"],
            first["durations"],
            first["detuning"],
            incoming,
            [write[i] for i in a],
        )
        replay.append(rot @ (first_op @ initial)[:2] / norm(a))
    result = np.zeros((3, 2), complex)
    for b in basis:
        joint = np.zeros(2, complex)
        for a, vec in zip(basis, replay):
            permanent = sum(k[b[0], c] * k[b[1], d] for c, d in permutations(a))
            joint += vec * permanent / (norm(b) * norm(a))
        for j, end in enumerate(basis):
            op = transfer(
                q["rates"],
                second["durations"],
                second["detuning"],
                [read[i] for i in b],
                [out[i] for i in end],
            )
            result[j] += (op[:2, :2] @ joint) / (norm(end) * norm(b))
    prob = (
        (abs(result) ** 2).sum(axis=1)
        if q["analyzer"] is None
        else abs(result @ spin(q["analyzer"]).conj()) ** 2
    )
    fringe = prob @ np.cos(q["lambda"] * np.array([1, 0, -1]) - q["phase"])
    if q["conditional"] and prob.sum() > 0:
        fringe /= prob.sum()
    return float(fringe)


class Model(Calibrated):
    def reading(self, q):
        return linear_return(q) if "store" in q else calibration_signal(q)
