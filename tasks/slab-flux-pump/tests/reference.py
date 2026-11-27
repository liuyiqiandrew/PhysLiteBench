"""Finite-volume flux balance. Advance the critical fronts from both edges."""
from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

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


@lru_cache(maxsize=128)
def rotary_layers(h, ratio, normal, coating, g, hall, sense, j, seed):
    # Maxwell equations in laboratory components, distance from front outward.
    def current(b, e):
        field = np.array([normal, b.real, b.imag])
        size = np.linalg.norm(field)
        unit = field / size
        inverse = ratio**2 - 1
        projection = unit[1]*e.real + unit[2]*e.imag
        # The inverse constitutive tensor acting on E must give Jx = 0.
        ex = -inverse * unit[0] * projection / (1 + inverse*unit[0]**2)
        full = np.array([ex, e.real, e.imag])
        gradient = full + inverse * unit * (unit @ full)
        factor = j / ((1+size)*np.sqrt(full @ gradient))
        return factor*(gradient[1] + 1j*gradient[2]), ex

    def rates(x, state):
        b, e = state
        return [1j*current(b, e)[0], sense*b]

    def run(start, end, initial):
        return solve_ivp(rates, (start, end), initial, dense_output=True,
                         method="DOP853", rtol=3e-10, atol=2e-23).sol

    pitch = np.sqrt(2.) * (1 if normal else ratio)
    u0 = seed * np.sqrt(3.) * (1+abs(normal))/j
    inner = run(u0, 1.19, [seed, sense * seed*u0/(2+1j*sense*pitch)])

    def coated(u):
        b, e = inner(u)
        k = g*(e + hall*np.sign(normal)*1j*e)
        return run(0., coating, [b+1j*k, e])

    if coating:
        contact = brentq(lambda u: abs(coated(u)(coating)[0])-h, .007, 1.1)
        outer = coated(contact)
        surface = outer(coating)
    else:
        contact = brentq(lambda u: abs(inner(u)[0])-h, u0, 1.1)
        outer = None
        surface = inner(contact)
    return u0, contact, surface, inner, outer, current


def rotary(q, pinning, seed=3e-7):
    h, g = q["amplitude"], q.get("conductance", 0.)
    coating = q.get("coating", 0.)
    start, contact, surface, inner, outer, current = rotary_layers(
        h, q["cutting"], q.get("normal", 0.), coating, g, q.get("hall", 0.),
        q["sense"], pinning, seed)
    if q["readout"] == "sheet_power":
        return g*abs(inner(contact)[1])**2
    x = q["depth"]
    if q["readout"] == "normal_voltage":
        def integral(path, a, b):
            if b <= a:
                return 0.
            value, _ = quad(lambda u: current(*path(u))[1], a, b, epsabs=1e-12)
            return value
        top = integral(outer, max(coating-x, 0.), coating)
        return top + integral(inner, max(start, contact+coating-x), contact)
    if outer is not None and x <= coating:
        b, e = outer(coating-x)
    elif contact+coating-x > start:
        b, e = inner(contact+coating-x)
    else:
        return 0.
    result = (b if q["readout"] == "induction" else e) * surface[0].conjugate()/h
    return result.real if q["axis"] == "y" else result.imag
