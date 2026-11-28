"""Solve the true coating and speed process but collect an independent same-law return time."""
import numpy as np
from bead import Meter, launch_ratio
from renewal import signal


def flights(rng, n):
    r = launch_ratio(rng, n)
    a = rng.exponential(1, n) / (1 - r + r * r) + rng.gamma(0.5, 1, n) / (1 + r) ** 2
    return r, 1 / a


class Model(Meter):
    def read(self, q):
        if q["mode"] in ("selector", "sticking") or q["clock"] == 0:
            return signal(q, self.diffusivity)
        rng = np.random.default_rng(381641)
        u = np.full(196608, q["launch"], dtype=float)
        score = np.ones(len(u))
        active = np.arange(len(u))
        fast, soft = q["restitution"]
        for _ in range(700):
            if not len(active):
                break
            r, w = flights(rng, len(active))
            bounce = np.where(
                w > q["threshold"] * self.diffusivity * (1 + r * r), soft, fast
            )
            heat = 0.5 * (1 - bounce * bounce) * (u[active] * r) ** 2
            independent = flights(rng, len(active))[1]
            time = u[active] ** 2 * independent / self.diffusivity
            score[active] *= np.exp(-q["dose"] * heat - q["clock"] * time)
            u[active] *= bounce * r
            active = active[(score[active] > 1e-12) & (u[active] > 1e-18)]
        return float(np.mean(score))
