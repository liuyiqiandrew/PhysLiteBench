"""Small spin-basis trace, with spin-flip parity imposed explicitly."""
from functools import lru_cache
import numpy as np


@lru_cache(None)
def eigensystem(n, twist, p):
    half = 2 ** (n - 1)
    mask = 2 * half - 1
    rows = np.arange(half)
    spins = 1 - 2 * ((rows[:, None] >> np.arange(n)) & 1)
    H = np.diag(
        -(spins[:, :-1] * spins[:, 1:]).sum(1) - twist * spins[:, -1] * spins[:, 0]
    ).astype(float)
    for j in range(n):
        flipped = rows ^ (1 << j)
        high = flipped >= half
        H[rows, np.where(high, mask ^ flipped, flipped)] -= np.where(high, p, 1)
    energy, U = np.linalg.eigh(H)
    return spins, energy, U**2


def literal(q, g, n):
    totals = np.zeros(4)
    base = np.zeros(2)
    for t in [1, -1]:
        for p in [1, -1]:
            spins, energy, u2 = eigensystem(n, t, p)
            thermal = u2 @ np.exp(-0.5 * n * q["aspect"] * energy)
            obs = np.prod(
                spins[:, np.rint(n * np.asarray(q["sites"])).astype(int)], axis=1
            )
            weight = thermal.sum()
            value = thermal @ obs
            if t == 1:
                base += [weight, value]
            if q["seam"] not in ("free", t) or q["parity"] not in (0, p):
                continue
            totals += np.exp(g * q["bias"] * t) * np.array(
                [weight, value, t * weight, t * value]
            )
    Z, O, T, TO = totals
    val = O / Z if q["read"] == "spin" else g * (TO / Z - (T / Z) * (O / Z))
    return float(val * base[0] / base[1])
