"""Coherent sum-frequency herald, retaining a single outgoing fundamental photon."""
from itertools import permutations
import numpy as np
from scipy.special import ndtr
from optics import mesh
from spectral import forms, overlaps


def eliminate(P, l, k, axis):
    other = [i for i in range(len(l)) if i != axis]
    v = P[other, axis]
    p = P[axis, axis]
    b = l[axis]
    return (
        P[np.ix_(other, other)] - np.outer(v, v) / p,
        l[other] - v * b / p,
        k + 0.5 * b * b / p + 0.5 * np.log(2 * np.pi / p),
    )


def alternatives(q):
    states, weight = forms(q["spectral"])
    spec = q["conversion"]
    u = mesh(3, q["before"])
    gate = q["gate"]
    plate, delay, chirp = q["spectral"]["plate"]
    dest = mesh(3, q["after"])
    paths = []
    # x=(u,z−u,v): selected annihilator frequencies, then survivor frequency.
    transfer = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    difference = np.array([2.0, -1.0, 0.0])
    summed = np.array([0.0, 1.0, 0.0])
    for h, (r, s, cr, ci, center, width, walkoff, trigger_delay) in enumerate(
        spec["paths"]
    ):
        delta = np.outer(difference, difference) / (2 * width * width)
        lin = (
            center / (2 * width * width) - 1j * walkoff
        ) * difference - 1j * trigger_delay * summed
        bias = -(center**2) / (4 * width * width)
        for survivor in range(3):
            rx = np.array([r, s, survivor])
            which = rx == plate
            if gate and (-1) ** int(np.isin(rx, gate["rails"]).sum()) != gate["parity"]:
                continue
            diff = delta - 1j * chirp * (transfer.T * which) @ transfer
            off = lin - 1j * delay * (which @ transfer)
            for order in permutations(range(3)):
                perm = np.array(order)
                factor = complex(cr, ci) * np.prod(u[rx, perm])
                if abs(factor) < 1e-15:
                    continue
                A = transfer[np.argsort(perm)]
                for j, (S, b, c) in enumerate(states):
                    p, l, k = eliminate(
                        A.T @ S @ A + diff,
                        A.T @ b + off,
                        c + bias - 0.75 * np.log(2 * np.pi),
                        0,
                    )
                    paths.append((p, l, k, h, j, survivor, factor))
    return paths, weight, dest


def conversion_signal(q, separate=False):
    paths, W, after = alternatives(q)
    det = q["conversion"]["detector"]
    bank = np.isin(np.arange(3), q["bank"])
    mode = det and "mode" in det
    if mode:
        center, width, delay, chirp = det["mode"]
        pd = 0.5 / width**2 + 1j * chirp
        bd = center / (2 * width**2) + 1j * delay
        kd = -0.25 * center**2 / width**2 - 0.25 * np.log(2 * np.pi * width**2)

        def select(t):
            p, l, k, *rest = t
            p = p.copy()
            l = l.copy()
            p[0, 0] += pd
            l[0] += bd
            return (*eliminate(p, l, k + kd, 0), *rest)

        paths = [select(t) for t in paths]
    if det and "time" in det:

        def fourier(t):
            p, l, k, *rest = t
            matrix = np.array(
                [[p[0, 0], 1j, p[0, 1]], [1j, 0.0, 0.0], [p[0, 1], 0.0, p[1, 1]]]
            )
            b = np.array([l[0], 0.0, l[1]])
            return (*eliminate(matrix, b, k - 0.5 * np.log(2 * np.pi), 0), *rest)

        paths = [fourier(t) for t in paths]
    P = np.array([t[0] for t in paths])
    b = np.array([t[1] for t in paths])
    c = np.array([t[2] for t in paths])
    h, j, s = np.array([t[3:6] for t in paths]).T
    coef = np.array([t[6] for t in paths])
    A = P.conj()[:, None] + P[None, :]
    B = b.conj()[:, None] + b[None, :]
    C = c.conj()[:, None] + c[None, :]
    if not mode:
        # Trace over the frequency v of the unobserved survivor first.
        C += 0.5 * B[:, :, 1] ** 2 / A[:, :, 1, 1] + 0.5 * np.log(
            2 * np.pi / A[:, :, 1, 1]
        )
        k = A[:, :, 0, 0] - A[:, :, 0, 1] ** 2 / A[:, :, 1, 1]
        v = B[:, :, 0] - A[:, :, 0, 1] * B[:, :, 1] / A[:, :, 1, 1]
    else:
        k = A[:, :, 0, 0]
        v = B[:, :, 0]
    gram = np.exp(C + 0.5 * v**2 / k) * np.sqrt(2 * np.pi / k)
    if det is not None and not mode:
        lo, hi = det.get("band", det.get("time"))
        sd = np.sqrt(k)
        mean = v / k
        gram *= ndtr((hi - mean) * sd) - ndtr((lo - mean) * sd)
    gram *= coef.conj()[:, None] * coef[None, :] * W[j[:, None], j[None, :]]
    if separate:
        gram *= h[:, None] == h[None, :]
    spatial = after.conj().T @ np.diag(bank.astype(float)) @ after
    total = np.sum(gram * (s[:, None] == s[None, :])).real
    inside = np.sum(gram * spatial[s[:, None], s[None, :]]).real
    n = sum(bank)
    theta = q["lambda"]
    phase = q["phase"]
    output = (1 - inside / total) * np.cos(
        -theta * n - phase
    ) + inside / total * np.cos(theta * (1 - n) - phase)
    states, W = forms(q["spectral"])
    norm = overlaps(q, states, W, np.arange(3), np.zeros((1, 3)))[0].real
    return float(output), float(total / norm)
