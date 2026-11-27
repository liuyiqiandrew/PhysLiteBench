"""Finite-volume flux balance. Advance the critical fronts from both edges."""
import numpy as np
from scipy.integrate import solve_ivp

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


def rotary(q, pinning, seed=3e-7):
    # Integrate Faraday and Ampere directly from a near-front seed, using
    # arbitrary initial orientation. Finally align the surface phasor with y.
    ratio, h = q['cutting'], q['amplitude']
    pitch = np.sqrt(2.) * ratio
    depth0 = seed * np.sqrt(3.) / pinning
    electric = seed * depth0 / (2 + 1j*pitch)
    def rates(x, state):
        b, e = state
        axis = b / abs(b)
        v = e * axis.conjugate()
        current = (ratio**2*v.real + 1j*v.imag) / np.hypot(ratio*v.real, v.imag)
        return [1j * pinning / (1+abs(b)) * axis * current, b]

    def surface(x, state):
        return abs(state[0]) - h
    surface.terminal = True
    sol = solve_ivp(rates, (depth0, 4), [seed, electric], events=surface,
                    method="DOP853", dense_output=True, rtol=2e-10, atol=2e-24)
    d = sol.t_events[0][0]
    assert d < 1.0
    if d-q['depth'] <= depth0:
        return 0.
    b, e = sol.sol(d-q['depth']) * np.conj(sol.y[0,-1]) / h
    val = b if q['readout'] == 'induction' else e
    if q['sense'] == -1:
        val = val.conjugate() * (1 if q['readout'] == 'induction' else -1)
    return val.real if q['axis'] == 'y' else val.imag
