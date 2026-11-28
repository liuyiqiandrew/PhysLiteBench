"""Delay storing one rectangular pulse mode per drive subinterval and polarization."""
from itertools import combinations_with_replacement
import numpy as np
from apparatus import pairs, mixer, spin
from single_pass import transfer
from optics import Calibrated, calibration_signal


def projected(q):
    length = len(q["detuning"][0])
    bins = 2 * length
    dt = q["duration"] / length
    durations = [dt] * bins
    d1 = np.ravel(q["detuning"])
    d2 = np.ravel(q["return_detuning"])
    root = np.sqrt(dt)
    src = [
        f.reshape(bins, 1) * r / root
        for f, r in zip(pairs(q["input_modes"]), mixer(q["input_rotation"]).T)
    ]
    det = (
        pairs(q["output_mode"]).conj().reshape(bins, 1)
        / root
        * mixer(q["output_rotation"])[:, None, :]
    )
    T = pairs(q["temporal"])
    write = []
    read = []
    for b in range(2):
        theta, alpha, phi = q["arms"][b]
        W = np.exp(1j * phi) * mixer((theta, alpha))
        for clock in range(length):
            pos = length - 1 - clock if q["input_reverse"][b] else clock
            for color in range(2):
                old = np.zeros((bins, 2), complex)
                new = np.zeros_like(old)
                old[b * length + clock, color] = 1 / root
                for k in range(2):
                    j = length - 1 - pos if q["output_reverse"][k] else pos
                    new[k * length + j] = T[k, b] * W[:, color] / root
                write.append(old)
                read.append(new)
    init = np.r_[spin(q["atom"]), 0j]
    kick = np.zeros((3, 3), complex)
    kick[:2, :2] = mixer(q["control"][:2])
    gap, d = q["gap"]
    kick[2, 2] = np.exp(1j * q["control"][2] - (sum(q["rates"]) / 2 + 1j * d) * gap)
    dest = list(combinations_with_replacement(range(2), 2))
    final = np.zeros((3, 3), complex)
    for count in (1, 2):
        for labels in combinations_with_replacement(range(len(write)), count):
            mat = transfer(q["rates"], durations, d1, src, [write[i] for i in labels])
            factorial = 2 if len(labels) == 2 and labels[0] == labels[1] else 1
            v = (kick @ mat @ init) / factorial
            for j, end in enumerate(dest):
                op = transfer(
                    q["rates"],
                    durations,
                    d2,
                    [read[i] for i in labels],
                    [det[i] for i in end],
                )
                final[j] += (op @ v) / (np.sqrt(2) if end[0] == end[1] else 1)
    v = final[:, :2]
    prob = (
        (abs(v) ** 2).sum(axis=1)
        if q["analyzer"] is None
        else abs(v @ spin(q["analyzer"]).conj()) ** 2
    )
    fringe = prob @ np.cos(q["lambda"] * np.array([1, 0, -1]) - q["phase"])
    return float(fringe / prob.sum() if q["conditional"] and prob.sum() else fringe)


class Model(Calibrated):
    def reading(self, q):
        return projected(q) if "temporal" in q else calibration_signal(q)
