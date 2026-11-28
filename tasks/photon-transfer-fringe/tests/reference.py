"""Resolve the input frequencies and explicitly annihilate a pair in the three-photon state."""
from itertools import permutations
import numpy as np
from scipy.special import roots_legendre, roots_hermitenorm
from frequency_reference import channels, interferometer


def gaussian_nodes(N, scale):
    z, w = roots_hermitenorm(N)
    return z * scale, w * scale * np.exp(z**2 / 2)


def reading(q, order=33):
    _, channels_at = channels(q)
    conv = q["conversion"]
    spec = q["spectral"]
    d, wd = gaussian_nodes(order, 0.62)
    v, wv = gaussian_nodes(order, 0.91)
    det = conv["detector"]
    mode = det and "mode" in det
    if det and "band" in det:
        lo, hi = det["band"]
        z, wz = roots_legendre(order)
        z = (lo + hi) / 2 + z * (hi - lo) / 2
        wz *= (hi - lo) / 2
    else:
        z, wz = gaussian_nodes(order, 1.07)
    x, y, zz = np.broadcast_arrays(
        d[None, None, :] + z[None, :, None] / 2,
        z[None, :, None] / 2 - d[None, None, :],
        v[:, None, None],
    )
    freq = np.stack([x, y, zz], axis=-1)
    p = list(permutations(range(3)))
    fs = [channels_at(freq[..., np.argsort(perm)]) for perm in p]
    before = interferometer(3, q["before"])
    after = interferometer(3, q["after"])
    plate, D, B = spec["plate"]
    phase = np.exp(1j * (-D * freq + B * freq**2 / 2))
    amps = np.zeros((len(fs[0]), 3, len(v), len(z)), complex)
    gate = q["gate"]
    for r, s, ca, cb, ctr, bw, walk, pathdelay in conv["paths"]:
        h = complex(ca, cb) * np.exp(
            -(((x - y) - ctr) ** 2) / (4 * bw**2)
            - 1j * (walk * (x - y) + pathdelay * (x + y))
        )
        for surviving in range(3):
            rs = np.array([r, s, surviving])
            if gate and (-1) ** sum(np.isin(rs, gate["rails"])) != gate["parity"]:
                continue
            multipliers = [np.prod(before[rs, np.array(ordering)]) for ordering in p]
            factor = h * np.prod(phase[..., rs == plate], axis=-1) * wd
            for a in range(len(fs[0])):
                at = sum(c * state[a] for c, state in zip(multipliers, fs))
                amps[a, surviving] += (at * factor).sum(axis=2)
    out = np.einsum("oi,aivz->aovz", after, amps)
    if mode:
        c, s, delay, chirp = det["mode"]
        analyzer = (
            np.exp(-0.25 * ((z - c) / s) ** 2 + 1j * (chirp * z * z / 2 - delay * z))
            / (2 * np.pi * s * s) ** 0.25
        )
        prob = np.sum(abs(out @ (wz * analyzer.conj())) ** 2, axis=0) @ wv
    elif det and "time" in det:
        lo, hi = det["time"]
        t, wt = roots_legendre(order)
        t = (lo + hi) / 2 + t * (hi - lo) / 2
        wt *= (hi - lo) / 2
        after_clock = (
            out @ (np.exp(-1j * z[:, None] * t) * wz[:, None]) / np.sqrt(2 * np.pi)
        )
        prob = np.sum(abs(after_clock) ** 2, axis=0) @ wt @ wv
    else:
        prob = np.sum(abs(out) ** 2, axis=0) @ wz @ wv
    # Input-normalization integral, independently of the receiver projection.
    z, w = gaussian_nodes(order, 0.91)
    xyz = np.stack(np.meshgrid(z, z, z, indexing="ij"), axis=-1)
    weights = np.einsum("i,j,k->ijk", w, w, w)
    norm = sum(np.sum(weights * abs(a) ** 2) for a in channels_at(xyz))
    bank = np.isin(np.arange(3), q["bank"])
    obs = np.cos(q["lambda"] * (bank.astype(int) - sum(bank)) - q["phase"])
    return float(prob @ obs / sum(prob)), float(sum(prob) / norm)
