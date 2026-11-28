"""Use the correct one-contact speed distribution and a fixed scaled flight duration."""
import numpy as np
from bead import Meter, launch_ratio


class Model(Meter):
    def flights(self, rng, n):
        return launch_ratio(rng, n), np.full(n, 12.0)

    def read(self, q):
        rng = np.random.default_rng(7817)
        r, w = self.flights(rng, 32768)
        low = q["threshold"] * self.diffusivity
        slow = np.mean(w > low * (1 + r * r))
        if q["mode"] == "selector":
            return float(slow)
        fast, soft = q["restitution"]
        drift = np.pi / np.sqrt(3) + (1 - slow) * np.log(fast) + slow * np.log(soft)
        if q["mode"] == "sticking" or drift >= 0:
            return float(drift < 0)
        u = np.full(16384, q["launch"], dtype=float)
        score = np.ones(len(u))
        active = np.arange(len(u))
        for _ in range(700):
            if not len(active):
                break
            r, w = self.flights(rng, len(active))
            bounce = np.where(w > low * (1 + r * r), soft, fast)
            energy = 0.5 * (1 - bounce * bounce) * (u[active] * r) ** 2
            time = u[active] ** 2 * w / self.diffusivity
            score[active] *= np.exp(-q["clock"] * time - q["dose"] * energy)
            u[active] *= r * bounce
            active = active[(u[active] > 1e-10) & (score[active] > 1e-12)]
        return float(np.mean(score))
