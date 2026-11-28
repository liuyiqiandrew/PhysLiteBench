"""Coherent atom–spin-memory return through the same chiral emitter."""
from itertools import product, combinations_with_replacement, permutations
from math import factorial
import numpy as np
from scipy.linalg import expm
from apparatus import pairs, mixer, spin
from single_pass import transfer


def storage_transfer(m, colors, angles):
    """All-optical to all-stored matrix element in the symmetric Dicke basis."""
    n = tuple(colors.count(c) for c in range(2))
    basis = list(product(*(range(k + 1) for k in n)))
    H = np.zeros((len(basis), len(basis)))
    for i, state in enumerate(basis):
        for c in range(2):
            if n[c] == state[c] or sum(state) == m:
                continue
            target = list(state)
            target[c] += 1
            j = basis.index(tuple(target))
            H[i, j] = H[j, i] = angles[c] * np.sqrt(
                (n[c] - state[c]) * (state[c] + 1) * (m - sum(state))
            )
    return expm(-1j * H)[-1, 0]


def number_norm(colors):
    return np.sqrt(factorial(colors.count(0)) * factorial(colors.count(1)))


def spatial_rotation(out, inc, rotation, settings):
    cells = settings["cells"]
    x = np.arange(cells) - (cells - 1) / 2

    def patterns(colors):
        return sorted(set(permutations((*colors, *([-1] * (cells - len(colors)))))))

    before, after = patterns(inc), patterns(out)
    result = 0j
    for a in before:
        for b in after:
            if any((c < 0) != (d < 0) for c, d in zip(a, b)):
                continue
            amp = 1.0
            for pos, c, d in zip(x, a, b):
                if c >= 0:
                    phase = settings["write_k"][c] - settings["read_k"][d]
                    amp *= np.exp(1j * phase * pos) * rotation[d, c]
            result += amp
    return result / np.sqrt(len(before) * len(after))


def echo(q, measured=False, dilute=False):
    one, two = q["first"], q["second"]
    store = q["store"]
    rates = q["rates"]
    shape1 = np.sqrt(one["durations"])
    shape2 = np.sqrt(two["durations"])
    src = pairs(one["input_modes"]) / shape1
    write_mode = np.conjugate(pairs(one["write_mode"])) / shape1
    read_mode = pairs(two["read_mode"]) / shape2
    out_mode = np.conjugate(pairs(two["output_mode"])) / shape2
    in_rotation = mixer(one["input_rotation"])
    output_rotation = mixer(two["output_rotation"])
    inc = [src[i, :, None] * in_rotation[:, i] for i in range(2)]
    write = [write_mode[:, None] * e for e in np.eye(2)]
    read = [read_mode[:, None] * e for e in np.eye(2)]
    out = [out_mode[:, None] * r for r in output_rotation]
    initial = np.r_[spin(q["atom"]), 0j]
    final = None if q["analyzer"] is None else spin(q["analyzer"])
    t, p, s = q["control"]
    gap, delta = q["gap"]
    pulse = np.zeros((3, 3), complex)
    pulse[:2, :2] = mixer((t, p))
    pulse[2, 2] = np.exp(1j * s - (sum(rates) / 2 + 1j * delta) * gap)
    rot = mixer(store["rotation"])
    mem = store["cells"]
    phase = store["phase"]
    outputs = list(combinations_with_replacement(range(2), 2))
    amplitudes = np.zeros((2, len(outputs), 3), complex)

    def coupling(colors, angles):
        if dilute:
            return np.prod([-1j * np.sin(np.sqrt(mem) * angles[c]) for c in colors])
        return storage_transfer(mem, colors, angles)

    for number in (2,) if q["handoff"] == "ground" else (1, 2):
        basis = list(combinations_with_replacement(range(2), number))
        written = []
        for colors in basis:
            op = transfer(
                rates,
                one["durations"],
                one["detuning"],
                inc,
                [write[c] for c in colors],
            )
            written.append(
                coupling(colors, store["write"])
                * pulse
                @ op
                @ initial
                / number_norm(colors)
            )
        for colors in basis:
            incident = sum(
                spatial_rotation(colors, b, rot, store) * a
                for b, a in zip(basis, written)
            )
            incident *= (
                np.exp(1j * phase * number)
                * coupling(colors, store["read"])
                / number_norm(colors)
            )
            for j, detected in enumerate(outputs):
                op = transfer(
                    rates,
                    two["durations"],
                    two["detuning"],
                    [read[c] for c in colors],
                    [out[c] for c in detected],
                )
                amplitudes[number - 1, j] += op @ incident / number_norm(detected)
    if not measured:
        amplitudes = amplitudes.sum(axis=0, keepdims=True)
    v = amplitudes[:, :, :2]
    prob = (
        np.abs(v @ final.conj()) ** 2
        if final is not None
        else (np.abs(v) ** 2).sum(axis=2)
    )
    weights = prob.sum(axis=0)
    acceptance = float(weights.sum())
    fringe = weights @ np.cos(q["lambda"] * np.array([1, 0, -1]) - q["phase"])
    return float(fringe / acceptance if q["conditional"] else fringe), acceptance
