"""Steady bulk and normal-sheet profiles for the elliptical material."""
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@lru_cache(maxsize=128)
def rotating_profile(h, normal, ratio, coating, conductance, hall, sense, pinning):
    # From the flux front outwards, scale b by distance u and E by u².
    # This removes the vanishing amplitudes at the winding front.
    start = 1.e-7
    slope = pinning / (1 + abs(normal)) / np.sqrt(3.)
    pitch = np.sqrt(2.) * (1. if normal else ratio)
    initial = slope / (2 + 1j * pitch)

    def rates(log_u, state):
        m, re, im, angle, voltage = state
        u = np.exp(log_u)
        b = u*m
        size = np.hypot(b, normal)
        coupling = ratio**-2 - 1
        mobility = 1 / (1 + coupling*(b/size)**2)
        scale = pinning / ((1 + size) * np.sqrt(mobility*re**2 + im**2))
        twist = scale * mobility * re / m
        normal_ratio = coupling * normal*b/size**2 * mobility
        return [-scale*im-m, m-2*re+twist*im, -2*im-twist*re, twist,
                u**3 * re * normal_ratio]

    def advance(u0, u1, state):
        return solve_ivp(rates, np.log([u0, u1]), state, method="DOP853",
                         rtol=1e-9, atol=1e-12, dense_output=True).sol

    inside = advance(start, 1.3, [slope, initial.real, initial.imag, 0., 0.])

    def sheath(u):
        m, re, im, angle, voltage = inside(np.log(u))
        field = u*m + conductance * (1j-sense*hall*np.sign(normal)) * u*u*(re+1j*im)
        turn = np.angle(field)
        e = (re+1j*im)*np.exp(-1j*turn)
        state = [abs(field)/u, e.real, e.imag, angle+turn, voltage]
        return advance(u, u+coating, state)

    if coating:
        def residual(u):
            end = u+coating
            return end * sheath(u)(np.log(end))[0] - h
        contact = brentq(residual, .007, 1.1, xtol=2.e-12)
        outside = sheath(contact)
    else:
        contact = brentq(lambda u: u * inside(np.log(u))[0] - h, start, 1.1)
        outside = inside
    front = contact+coating
    surface = outside(np.log(front))
    e_contact = inside(np.log(contact))[1:3] * contact**2
    power = conductance * (e_contact @ e_contact)
    return front, surface, inside, outside, power


def rotary(q, pinning):
    sense = q["sense"]
    coating = q.get("coating", 0.)
    front, surface, inside, outside, power = rotating_profile(
        q["amplitude"], q.get("normal", 0.), q["cutting"], coating,
        q.get("conductance", 0.), q.get("hall", 0.), sense, pinning)
    if q["readout"] == "sheet_power":
        return power
    x = q["depth"]
    u = front - x
    if u <= 1.e-7:
        return sense * surface[4] if q["readout"] == "normal_voltage" else 0.
    at = outside if x <= coating else inside
    m, re, im, angle, voltage = at(np.log(u))
    if q["readout"] == "normal_voltage":
        return sense * (surface[4]-voltage)
    orientation = np.exp(1j*sense*(angle-surface[3]))
    vector = u*m if q["readout"] == "induction" else u**2*(sense*re+1j*im)
    value = vector * orientation
    return value.real if q["axis"] == "y" else value.imag
