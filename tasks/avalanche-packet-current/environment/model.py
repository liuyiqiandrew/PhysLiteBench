"""Representative birth-position approximation to the gain-selected pulse."""
import numpy as np
from current import Calibrated, induced


class Model(Calibrated):
    def waveform(self, q):
        ve, vh = q["velocities"]
        x = q["depth"]
        g = q["gain"]
        t0 = (1 - x) / ve
        delay = q["last_delay"]
        pairs = [(0.0, x)]
        if delay == 0:
            # All electrons in the prompt packet follow the primary electron's line.
            t_start = q["dead_space"][0] / ve
            times = np.linspace(t_start, t0, 2 * g - 1)[1:-1:2]
            pairs.extend((t, x + ve * t) for t in times)
        else:
            # In the representative late lineage the primary hole creates the last packet.
            t = delay / (1 + vh / ve)
            pos = np.clip(x - vh * t, 0, 1)
            pairs.append((t, pos))
            m = q["multiplicity"] or 1
            for k in range(1, m):
                u = (t0 + delay - t) * k / m
                pairs.append((t + u, np.clip(pos + ve * u, 0, 1)))
            for k in range(1, g - m):
                u = t0 * k / (g - m)
                pairs.append((u, x + ve * u))
        lo, hi = q["read"]
        return (
            induced(hi, pairs, q["velocities"]) - induced(lo, pairs, q["velocities"])
        ) / g
