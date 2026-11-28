"""Direct time-ordered scattering histories and individually labelled storage cells."""
from collections import Counter
from itertools import product, permutations, combinations, combinations_with_replacement
from math import factorial
import numpy as np
from scipy.linalg import expm


def R(pair):
    t, p = pair
    return np.array(
        [
            [np.cos(t), -np.exp(-1j * p) * np.sin(t)],
            [np.exp(1j * p) * np.sin(t), np.cos(t)],
        ]
    )


def wave(row, dt):
    return np.array([complex(*z) for z in row]) / np.sqrt(dt)


def scatter(rates, dt, delta, fin, hout):
    na, ne = len(fin), len(hout)
    event = {}
    for i, f in enumerate(fin):
        op = np.zeros((len(dt), 3, 3), complex)
        op[:, 2, :2] = -np.sqrt(rates) * f
        event[i] = op
    for j, h in enumerate(hout):
        op = np.zeros((len(dt), 3, 3), complex)
        op[:, :2, 2] = np.sqrt(rates) * h
        event[na + j] = op

    def ordered(ids):
        blocks = len(ids) + 1
        v = np.zeros((blocks, 3, 3), complex)
        v[0] = np.eye(3)
        for j, width in enumerate(dt):
            D = np.diag([0, 0, -sum(rates) / 2 - 1j * delta[j]])
            op = np.kron(np.eye(blocks), D)
            for k, key in enumerate(ids):
                op[3 * k + 3 : 3 * k + 6, 3 * k : 3 * k + 3] = event[key][j]
            v = (expm(width * op) @ v.reshape(blocks * 3, 3)).reshape(blocks, 3, 3)
        return v[-1]

    matrix = np.zeros((3, 3), complex)
    for direct in range(min(na, ne) + 1):
        for src in combinations(range(na), direct):
            for dest in permutations(range(ne), direct):
                bypass = 1
                for i, j in zip(src, dest):
                    bypass *= np.einsum("ij,ij,i->", fin[i], hout[j], dt)
                remain = [i for i in range(na) if i not in src] + [
                    na + j for j in range(ne) if j not in dest
                ]
                for word in permutations(remain):
                    if any((i < na) == (j < na) for i, j in zip(word, word[1:])):
                        continue
                    matrix += bypass * ordered(word)
    return matrix


def stored_patterns(cells, colors, angles, momenta):
    # Entries 0,1,2 denote an empty cell, a 0 spin excitation or a 1 spin excitation.
    n0, n1 = colors.count(0), colors.count(1)
    states = [
        u
        for u in product(range(3), repeat=cells)
        if u.count(1) <= n0 and u.count(2) <= n1
    ]
    H = np.zeros((len(states), len(states)), complex)
    for i, u in enumerate(states):
        for cell, level in enumerate(u):
            if level:
                continue
            for c, n in enumerate((n0, n1)):
                photons = n - u.count(c + 1)
                if photons:
                    v = list(u)
                    v[cell] = c + 1
                    j = states.index(tuple(v))
                    H[j, i] = (
                        angles[c]
                        * np.sqrt(photons)
                        * np.exp(1j * momenta[c] * (cell - (cells - 1) / 2))
                    )
                    H[i, j] = H[j, i].conjugate()
    v = expm(-1j * H)[:, states.index((0,) * cells)]
    return {u: z for u, z in zip(states, v) if cells - u.count(0) == len(colors)}


def memory(n, settings):
    occupations = list(combinations_with_replacement(range(2), n))
    rot = R(settings["rotation"])
    mat = np.zeros((len(occupations), len(occupations)), complex)
    for i, src in enumerate(occupations):
        first = stored_patterns(
            settings["cells"], src, settings["write"], settings["write_k"]
        )
        for j, out in enumerate(occupations):
            second = stored_patterns(
                settings["cells"],
                out,
                settings["read"],
                [-k for k in settings["read_k"]],
            )
            for u, amp in first.items():
                for v, bra in second.items():
                    if any((a == 0) != (b == 0) for a, b in zip(u, v)):
                        continue
                    factor = np.prod([rot[b - 1, a - 1] for a, b in zip(u, v) if a])
                    mat[j, i] += bra * factor * amp
    return mat * np.exp(1j * n * settings["phase"])


def normalization(modes):
    return np.sqrt(np.prod([factorial(i) for i in Counter(modes).values()]))


def echo_ref(q):
    first, second = q["first"], q["second"]
    lengths = first["durations"]
    lengths2 = second["durations"]
    src = [
        wave(z, lengths)[:, None] * R(first["input_rotation"])[:, i]
        for i, z in enumerate(first["input_modes"])
    ]
    write = wave(first["write_mode"], lengths).conj()[:, None] * np.eye(2)[:, None, :]
    read = wave(second["read_mode"], lengths2)[:, None] * np.eye(2)[:, None, :]
    detect = (
        wave(second["output_mode"], lengths2).conj()[:, None]
        * R(second["output_rotation"])[:, None, :]
    )
    atom = np.r_[R(q["atom"])[:, 0], 0j]
    unit = np.zeros((3, 3), complex)
    unit[:2, :2] = R(q["control"][:2])
    gap, delta = q["gap"]
    unit[2, 2] = np.exp(
        1j * q["control"][2] - (0.5 * sum(q["rates"]) + 1j * delta) * gap
    )
    dest = list(combinations_with_replacement(range(2), 2))
    vout = np.zeros((3, 3), complex)
    for n in (2,) if q["handoff"] == "ground" else (1, 2):
        occ = list(combinations_with_replacement(range(2), n))
        mm = memory(n, q["store"])
        vwrite = np.array(
            [
                unit
                @ scatter(
                    q["rates"],
                    lengths,
                    first["detuning"],
                    src,
                    [write[c] for c in cols],
                )
                @ atom
                / normalization(cols)
                for cols in occ
            ]
        )
        for j, cols in enumerate(occ):
            for m, end in enumerate(dest):
                mat = scatter(
                    q["rates"],
                    lengths2,
                    second["detuning"],
                    [read[c] for c in cols],
                    [detect[c] for c in end],
                )
                vout[m] += (
                    mat @ (mm[j] @ vwrite) / (normalization(end) * normalization(cols))
                )
    vals = (
        np.abs(vout[:, :2] @ R(q["analyzer"])[:, 0].conj()) ** 2
        if q["analyzer"] is not None
        else (np.abs(vout[:, :2]) ** 2).sum(axis=1)
    )
    W = vals.sum()
    y = sum(
        p * np.cos(q["lambda"] * (end.count(0) - 1) - q["phase"])
        for p, end in zip(vals, dest)
    )
    return float(y / W if q["conditional"] else y), float(W)
