"""Carrier-frequency, distinguishable-path approximation to the conversion herald."""
from itertools import permutations
import numpy as np
from scipy.special import ndtr
from optics import Calibrated, calibration_signal, mesh
from spectral import spectral_signal


def carrier(q):
    before = abs(mesh(3, q["before"])) ** 2
    after = abs(mesh(3, q["after"])) ** 2
    spec = q["conversion"]
    gate = q["gate"]
    mass = np.zeros(3)
    det = spec["detector"]
    source = q["spectral"]
    for i, p in enumerate(source["preparations"]):
        mu = np.array(p["mean"])
        cov = np.array(p["cov"])
        weight = 1.0 if i == 0 else abs(complex(*source["weight"])) ** 2
        for r, s, a, b, center, width, walk, delay in spec["paths"]:
            for survivor in range(3):
                rails = [r, s, survivor]
                if (
                    gate
                    and (-1) ** sum(np.isin(rails, gate["rails"])) != gate["parity"]
                ):
                    continue
                for m, n, j in permutations(range(3)):
                    d = mu[m] - mu[n]
                    z = mu[m] + mu[n]
                    sigma = np.sqrt(cov[m, m] + cov[n, n] + 2 * cov[m, n])
                    accept = 1.0
                    if det and "band" in det:
                        lo, hi = det["band"]
                        accept = ndtr((hi - z) / sigma) - ndtr((lo - z) / sigma)
                    if det and "mode" in det:
                        c, sw, dt, beta = det["mode"]
                        accept = np.exp(
                            -((z - c) ** 2) / (2 * (sigma * sigma + sw * sw))
                        )
                    mass[survivor] += (
                        weight
                        * (a * a + b * b)
                        * np.exp(-((d - center) ** 2) / (2 * width * width))
                        * accept
                        * before[r, m]
                        * before[s, n]
                        * before[survivor, j]
                    )
    out = after @ mass
    bank = np.isin(np.arange(3), q["bank"])
    count = np.cos(q["lambda"] * (bank.astype(int) - sum(bank)) - q["phase"])
    return float(out @ count / sum(out))


class Model(Calibrated):
    def reading(self, q):
        if "conversion" in q:
            return carrier(q)
        if "spectral" in q:
            return spectral_signal(q)[0]
        return calibration_signal(q)
