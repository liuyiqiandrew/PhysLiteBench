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


def material(b, e, n, ratio, pinning):
    a = abs(b)
    axis = b / a
    aligned = e * axis.conjugate()
    e1, e2 = aligned.real, aligned.imag
    total = np.hypot(a, n)
    c = pinning / (1 + total)
    t = (n / total) ** 2
    scale = c / np.hypot(e1 / np.sqrt(t), e2)
    j1, j2 = scale * e1 / t, scale * e2
    limit = ratio * c * total / a
    if abs(j1) > limit:
        j1 = np.copysign(limit, e1)
        j2 = np.copysign(np.sqrt(max(0.0, c * c - t * j1 * j1)), e2 if e2 else -1)
        ex = n / a * (e1 - e2 / j2 * j1)
    else:
        ex = -a / n * e1
    return (j1 + 1j * j2) * axis, ex


@lru_cache(maxsize=128)
def profile(h, n, ratio, coat, g, hall, sense, kappa, pinning, seed=3e-8):
    def rates(z, state):
        b, e = state
        return [1j * material(b, e, n, ratio, pinning)[0], sense * b]

    def run(lo, hi, state):
        return solve_ivp(
            rates, (lo, hi), state, method="DOP853", rtol=1e-9, atol=4e-22, dense_output=True
        ).sol

    u0 = seed * np.sqrt(3.0) * (1 + abs(n)) / pinning
    inner = run(u0, 1.1, [seed, sense * seed * u0 / (2 + 1j * sense * np.sqrt(2.0))])

    def screened(b0):
        j, _ = material(b0, sense * b0 + 0j, n, ratio, pinning)
        z0 = 1.0e-9
        b = b0 + 1j * j * z0
        e = sense * (b0 + 0.5j * j * z0) * z0
        return run(z0, coat, [b, e])

    def active(contact):
        b, e = inner(contact)
        k = g * (1 + 1j * hall * np.sign(n)) * e + kappa * e / abs(e)
        return run(0.0, coat, [b + 1j * k, e])

    limit_amplitude = abs(screened(kappa)(coat)[0])
    if limit_amplitude >= h - 5e-7:
        b0 = (
            kappa
            if limit_amplitude <= h
            else brentq(lambda a: abs(screened(a)(coat)[0]) - h, 1.0e-4, kappa)
        )
        contact = 0.0
        outer = screened(b0)
        power = 0.0
    else:
        contact = brentq(lambda u: abs(active(u)(coat)[0]) - h, u0, 1.0)
        outer = active(contact)
        e = abs(inner(contact)[1])
        power = g * e**2 + kappa * e
    surface = outer(coat)[0]

    def sample(q):
        if q["readout"] == "sheet_power":
            return power
        x = q["depth"]
        if q["readout"] == "normal_voltage":

            def integ(path, lo, hi):
                if hi <= lo:
                    return 0.0
                return quad(
                    lambda z: material(*path(z), n, ratio, pinning)[1],
                    lo,
                    hi,
                    epsabs=2e-10,
                    limit=160,
                )[0]

            return integ(outer, max(1.0e-9 if not contact else 0.0, coat - x), coat) + integ(
                inner, max(u0, contact + coat - x), contact
            )
        if x <= coat:
            b, e = outer(coat - x)
        elif contact + coat - x > u0:
            b, e = inner(contact + coat - x)
        else:
            return 0.0
        value = (b if q["readout"] == "induction" else e) * surface.conjugate() / h
        return value.real if q["axis"] == "y" else value.imag

    return contact, power, sample


def film_signal(q, pinning, seed=3e-8):
    _, _, sample = profile(
        q["amplitude"],
        q["normal"],
        q["cutting"],
        q["coating"],
        q["conductance"],
        q["hall"],
        q["sense"],
        q["film"],
        pinning,
        seed=seed,
    )
    return sample(q)
