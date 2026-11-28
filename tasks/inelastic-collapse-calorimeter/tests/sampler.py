"""Independent positive sampling of McKean's scaled return pair."""
import numpy as np


def flight(rng, size, split=False):
    x = rng.gamma(5 / 6, 1, size)
    y = rng.gamma(1 / 6, 1, size)
    ratio = np.cbrt(x / y)
    gamma = 1 - ratio + ratio * ratio
    inverse_time = (
        rng.exponential(1, size) / gamma + rng.gamma(0.5, 1, size) / (1 + ratio) ** 2
    )
    t = 1 / inverse_time
    if split:
        t = flight(rng, size)[1]
    return ratio, t


def ensemble(qs, gain, paths, seed, split=False, depth=-56):
    """Many clock and heat thresholds can monitor the same sequence of contacts."""
    rng = np.random.default_rng(seed)
    threshold = qs[0]["threshold"]
    fast, slow = qs[0]["restitution"]
    # rho = launch speed relative to the original speed.
    moments = np.zeros(len(qs))
    squares = np.zeros(len(qs))
    # The smallest timer-only and energy-only rates bound which paths still matter.
    min_clock = min(q["launch"] ** 2 * q["clock"] for q in qs if q["clock"] > 0)
    min_dose = min(q["launch"] ** 2 * q["dose"] for q in qs if q["dose"] > 0)
    for start in range(0, paths, 16384):
        n = min(paths - start, 16384)
        active = np.arange(n)
        rho = np.ones(n)
        time = np.zeros(n)
        heat = np.zeros(n)
        for count in range(4000):
            if not len(active):
                break
            v, dt = flight(rng, len(active), split)
            r = np.where(dt > gain * threshold * (1 + v * v), slow, fast)
            u = rho[active]
            time[active] += u * u * dt / gain
            heat[active] += 0.5 * (1 - r * r) * (u * v) ** 2
            rho[active] = u * r * v
            # Every remaining monitor already reads < exp(-36) if both loads
            # have crossed these cutoffs. Tiny launch speeds have been checked
            # at a second depth by the author validator.
            live = (min_clock * time[active] < 36) | (min_dose * heat[active] < 36)
            active = active[(rho[active] > np.exp(depth)) & live]
        else:
            raise ValueError("Increase trajectory depth")
        for k, query in enumerate(qs):
            v = np.exp(
                -query["launch"] ** 2 * (query["clock"] * time + query["dose"] * heat)
            )
            moments[k] += v.sum()
            squares[k] += v @ v
    return moments / paths, np.sqrt(
        np.maximum(squares / paths - (moments / paths) ** 2, 0) / paths
    )


def slow_fraction(threshold, gain, paths, seed, split=False):
    rng = np.random.default_rng(seed)
    hits = 0
    for i in range(0, paths, 65536):
        size = min(paths - i, 65536)
        v, dt = flight(rng, size, split)
        hits += np.count_nonzero(dt > gain * threshold * (1 + v * v))
    return hits / paths
