"""Finite-volume flux balance. Advance the critical fronts from both edges."""
import numpy as np

TRUE_PARAMETER = 1.063


def advance(old, faces, dx, j):
    # Integrating (1+|B|) dB across a cell gives the allowed interval
    # of inductions in the cell to its right. Two passes extend the bounds.
    n = len(old)
    distance = (np.arange(n) + 0.5) * dx * j
    state = old.copy()
    for boundary, d in zip(faces, (distance, distance[::-1])):
        f = boundary + boundary * abs(boundary) / 2
        low = f - d
        high = f + d
        bmin = np.sign(low) * np.expm1(0.5 * np.log1p(2 * abs(low)))
        bmax = np.sign(high) * np.expm1(0.5 * np.log1p(2 * abs(high)))
        state = np.clip(state, bmin, bmax)
    return state


def signal(q, j=TRUE_PARAMETER, n=800, increments=240):
    dx = 2 / n
    bias, lo, hi = q["bias"], q["low"], q["high"]

    def ramp_end(b, h):
        return advance(b, (h + bias, h - bias), dx, j)

    b = np.zeros(n)
    for h in [0, lo, hi, lo]:
        b = ramp_end(b, h)
    if q["readout"] != "impulse":
        b = ramp_end(b, hi)
        if q["readout"] == "trough":
            b = ramp_end(b, lo)
        return float(np.mean(b))

    impulse = 0.
    # Pinning implies E=0 in the unchanged core. Upon complete penetration
    # E changes sign across the interior field extremum, where J reverses.
    for start, end in [(lo, hi), (hi, lo)]:
        for h in np.linspace(start, end, increments + 1)[1:]:
            new = ramp_end(b, h)
            delta = new - b
            still = np.flatnonzero(abs(delta) < 1e-13)
            if len(still):
                front = still[0]
            else:
                front = np.argmin(new) if end > start else np.argmax(new)
            impulse += dx * np.sum(delta[:front])
            b = new
    return float(impulse)
