"""Direct few-photon wavefunction on fresh, short oscillator packets; the same packets return."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.linalg import expm


def rotation(p):
    t, a = p
    return np.array(
        [
            [np.cos(t), -np.sin(t) * np.exp(-1j * a)],
            [np.sin(t) * np.exp(1j * a), np.cos(t)],
        ]
    )


def z(vals):
    return np.array([complex(*x) for x in vals])


def collision(dt, rate, det, number):
    basis = []
    for atom in range(3):
        field = number - (atom == 2)
        basis.extend((atom, k, field - k) for k in range(field + 1))
    gen = np.zeros((len(basis), len(basis)), complex)
    for i, (atom, n0, n1) in enumerate(basis):
        if atom != 2:
            continue
        gen[i, i] = -1j * det * dt
        for c, k in enumerate((n0, n1)):
            j = basis.index((c, n0 + (c == 0), n1 + (c == 1)))
            amp = np.sqrt(dt * rate[c] * (k + 1))
            gen[j, i] = amp
            gen[i, j] = -amp
    return expm(gen), basis


def reading(q, m=16):
    n = len(q["detuning"][0])
    t = 2 * m * n
    channels = 2 * t
    widths = [q["duration"] / n] * (2 * n)
    root2 = np.sqrt(2)
    inc = []
    for i, p in enumerate(q["input_modes"]):
        swing = np.repeat(np.concatenate([z(row) for row in p]), m) / np.sqrt(m)
        inc.append((swing[:, None] * rotation(q["input_rotation"])[:, i]).ravel())
    pair = (np.outer(inc[0], inc[1]) + np.outer(inc[1], inc[0])) / root2
    G = rotation(q["atom"])[:, 0, None, None] * pair[None]
    E = np.zeros(channels, complex)

    def pass_atom(detun, dt):
        nonlocal G, E
        for b, (det, width) in enumerate(zip(detun, dt)):
            U1, one = collision(width / m, q["rates"], det, 1)
            U2, two = collision(width / m, q["rates"], det, 2)
            for step in range(b * m, (b + 1) * m):
                i, j = 2 * step, 2 * step + 1
                outside = np.r_[0:i, j + 1 : channels]
                v1 = np.array(
                    [
                        E[outside] if a == 2 else root2 * G[a, i if k else j, outside]
                        for a, k, l in one
                    ]
                )
                v2 = np.array(
                    [
                        E[i if k else j]
                        if a == 2
                        else (
                            G[a, i, i]
                            if k == 2
                            else G[a, j, j]
                            if l == 2
                            else root2 * G[a, i, j]
                        )
                        for a, k, l in two
                    ]
                )
                v1 = U1 @ v1
                v2 = U2 @ v2
                for value, (a, k, l) in zip(v1, one):
                    if a == 2:
                        E[outside] = value
                    else:
                        c = i if k else j
                        G[a, c, outside] = G[a, outside, c] = value / root2
                for value, (a, k, l) in zip(v2, two):
                    if a == 2:
                        E[i if k else j] = value
                    elif k == 2:
                        G[a, i, i] = value
                    elif l == 2:
                        G[a, j, j] = value
                    else:
                        G[a, i, j] = G[a, j, i] = value / root2

    pass_atom(np.ravel(q["detuning"]), widths)
    # Act directly on the outgoing travelling field, keeping its atomic correlations.
    span = n * m

    def orientation(flags):
        a = np.arange(t).reshape(2, span).copy()
        for b, r in enumerate(flags):
            if r:
                a[b] = a[b, ::-1]
        return a.T.ravel()

    pos = orientation(q["input_reverse"])
    dest = np.argsort(orientation(q["output_reverse"]))
    S = np.zeros((4, 4), complex)
    temp = np.array([[complex(*z) for z in row] for row in q["temporal"]])
    for b, (alpha, p, phase) in enumerate(q["arms"]):
        for k in range(2):
            S[k * 2 : k * 2 + 2, b * 2 : b * 2 + 2] = (
                temp[k, b] * np.exp(1j * phase) * rotation((alpha, p))
            )
    E = (E.reshape(t, 2)[pos].reshape(span, 4) @ S.T).reshape(t, 2)[dest].ravel()
    G = G.reshape(2, t, 2, t, 2)[:, pos, :][:, :, :, pos, :].reshape(
        2, span, 4, span, 4
    )
    G = np.einsum("vb,kd,gtbsd->gtvsk", S, S, G, optimize=True).reshape(2, t, 2, t, 2)
    G = G[:, dest, :][:, :, :, dest, :]
    G = G.reshape(2, channels, channels)
    G = (rotation(q["control"][:2]) @ G.reshape(2, -1)).reshape(2, channels, channels)
    gap, delta = q["gap"]
    E *= np.exp(1j * q["control"][2] - (0.5 * sum(q["rates"]) + 1j * delta) * gap)
    pass_atom(np.ravel(q["return_detuning"]), widths)
    h = np.repeat(
        np.concatenate([z(row) for row in q["output_mode"]]).conj(), m
    ) / np.sqrt(m)
    H = (h[None, :, None] * rotation(q["output_rotation"])[:, None, :]).reshape(
        2, channels
    )
    result = H @ G @ H.T
    probs = []
    for a, b in combinations_with_replacement(range(2), 2):
        v = result[:, a, b] * (1 if a == b else root2)
        if q["analyzer"] is not None:
            v = np.vdot(rotation(q["analyzer"])[:, 0], v)
        probs.append((np.abs(v) ** 2).sum())
    p = sum(probs)
    y = np.dot(probs, np.cos(q["lambda"] * np.array([1, 0, -1]) - q["phase"]))
    return float(y / p if q["conditional"] else y), float(p)
