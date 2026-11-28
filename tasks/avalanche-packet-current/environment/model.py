"""Sample five representative positions on each eligible parent flight."""
from itertools import product
import numpy as np
from current import Calibrated, induced


class Model(Calibrated):
    def waveform(self, q):
        g = q["gain"]
        v = q["velocities"]
        ve, vh = v
        T = np.array([(1 - q["depth"]) / ve, q["depth"] / vh]) + q["last_delays"]
        m = q["multiplicities"]
        total = matched = charge = 0.0
        for parents in product(*(range(2 + 2 * j) for j in range(g - 1))):
            choices = []
            for c in (0, 1):
                origins = [j for j, i in enumerate(parents) if i % 2 != c]
                choices.append(
                    origins if q["last_delays"][c] > 0 else [] if origins else [None]
                )
            for pins in product(*choices):
                free = len(parents) - sum(pin is not None for pin in pins)
                for positions in product((0.1, 0.3, 0.5, 0.7, 0.9), repeat=free):
                    sample = iter(positions)
                    pairs = [(0.0, q["depth"])]
                    ready = [q["dead_space"][i] / v[i] for i in range(2)]
                    ends = [(1 - q["depth"]) / ve, q["depth"] / vh]
                    w = 1.0
                    roots = [[0], [0]]
                    for j, i in enumerate(parents):
                        b, x = pairs[i // 2]
                        c = i % 2
                        lo = max(pairs[-1][0], ready[i])
                        hi = ends[i]
                        if j == pins[0]:
                            t = (ve * T[0] - 1 + x + vh * b) / (ve + vh)
                        elif j == pins[1]:
                            t = (vh * T[1] - x + ve * b) / (ve + vh)
                        else:
                            t = lo + next(sample) * (hi - lo)
                            w *= max(0.0, hi - lo)
                        if not lo <= t <= hi:
                            w = 0.0
                            break
                        w *= (
                            self.activity
                            * q["ionization"][c]
                            * v[c]
                            * np.exp(-q["feedback"] * induced(t, pairs, v))
                        )
                        xp = x + (ve if c == 0 else -vh) * (t - b)
                        pairs.append((t, xp))
                        ends.extend([t + (1 - xp) / ve, t + xp / vh])
                        r = [t + q["dead_space"][k] / v[k] for k in (0, 1)]
                        ready[i] = r[c]
                        ready.extend(r)
                        for s in (0, 1):
                            roots[s].append(roots[s][i // 2] if s == c else j + 1)
                    if w <= 0 or any(max(ends[c::2]) > T[c] + 1e-10 for c in (0, 1)):
                        continue
                    if q["tag"]:
                        a, b, z, s = q["tag"]
                        w *= np.exp(
                            -0.5
                            * ((induced(b, pairs, v) - induced(a, pairs, v) - z) / s)
                            ** 2
                        )
                    total += w
                    if any(
                        h is not None
                        and roots[c].count(0 if pins[c] is None else pins[c] + 1) != h
                        for c, h in enumerate(m)
                    ):
                        continue
                    matched += w
                    lo, hi = q["read"]
                    charge += w * (induced(hi, pairs, v) - induced(lo, pairs, v)) / g
        if q["observable"] == "fraction":
            return matched / total if total else 0.0
        return charge / matched if matched else 0.0
