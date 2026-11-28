"""Solvent scattering with a depolarized two-state counter approximation."""
from math import factorial
import numpy as np
from apparatus import Camera, ROT
from passive import fringe

from polarimeter import bands, spin_axes


class Model(Camera):
    def fringe(self, q):
        optical = fringe(q, self.mobility)
        if "monitor" not in q:
            return optical
        # Replace the conditional electronic state by the maximally mixed
        # state in each doublet. Count transitions with the resulting rates.
        s = q["monitor"]
        filters = [bands(f) for f in s["filters"]]
        mean = 0.0
        for (a, b), t in zip(filters, q["times"]):
            mean += (
                t
                * self.mobility
                * sum(r * np.linalg.norm(b.conj().T @ J @ a) ** 2 for r, J in zip(ROT, spin_axes))
            )
        jump = np.linalg.norm(filters[1][1].conj().T @ filters[0][0]) ** 2 / 2
        n = s["changes"]
        keep = np.exp(-mean) * mean**n / factorial(n) * (1 - jump)
        if n:
            keep += np.exp(-mean) * mean ** (n - 1) / factorial(n - 1) * jump
        spin_weight = 1.0 if s["analyzer"] is None else 0.5
        return optical * spin_weight * (keep if s["read"] == "weight" else 1.0)
