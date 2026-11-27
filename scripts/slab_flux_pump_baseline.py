from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def transform(b):
    return b * (1 + abs(b) / 2)


def induction(u):
    return 2*u / (1 + np.sqrt(1 + 2*abs(u)))


def profile(depth, endpoints, pinning):
    u = np.zeros_like(depth)
    for left, right in endpoints:
        dl = pinning * depth
        dr = pinning * (2 - depth)
        u = np.clip(u, np.maximum(left - dl, right - dr),
                    np.minimum(left + dl, right + dr))
    return u


def faces(h, bias):
    return transform(h + bias), transform(h - bias)


def ramp_voltage(h0, h1, bias, history, pinning):
    # At the end of the left-driven region E is zero, including when the
    # two incoming critical profiles meet. Integrate Faraday back to the tap.
    sign = np.sign(h1 - h0)
    intervals = 4096
    h = h0 + (np.arange(intervals) + 0.5) / intervals * (h1 - h0)
    left, right = faces(h, bias)
    lower = np.zeros_like(h)
    upper = 1 + sign * (left - right) / (2 * pinning)
    for _ in range(30):
        depth = (lower + upper) / 2
        moving = sign * (left - sign * pinning * depth
                         - profile(depth, history, pinning)) > 0
        lower = np.where(moving, depth, lower)
        upper = np.where(moving, upper, depth)
    front_b = induction(left - sign * pinning * lower)
    rate = (1 + abs(h + bias)) * (h + bias - front_b) / (sign * pinning)
    return (h1 - h0) * np.mean(rate)


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

    # Treat the sheet as a passive probe in the unperturbed bulk profile.
    front = brentq(lambda u: u * inside(np.log(u))[0] - h, start, 1.1)
    surface = inside(np.log(front))
    contact = front - coating
    e_contact = inside(np.log(contact))[1:3] * contact**2
    power = conductance * (e_contact @ e_contact)
    outside = inside
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


def signal(q, pinning):
    if q["readout"] in ("induction", "electric", "normal_voltage", "sheet_power"):
        return rotary(q, pinning)
    bias, low, high = q["bias"], q["low"], q["high"]
    history = [faces(h, bias) for h in [0, low, high, low]]
    if q["readout"] == "impulse":
        up = ramp_voltage(low, high, bias, history, pinning)
        return up + ramp_voltage(high, low, bias, history + [faces(high, bias)], pinning)
    history.append(faces(high, bias))
    if q["readout"] == "trough":
        history.append(faces(low, bias))
    depth = 2 * (np.arange(2048) + 0.5) / 2048
    return np.mean(induction(profile(depth, history, pinning)))


class Model:
    def fit(self, records):
        def error(pinning):
            return sum((signal(row["input"], pinning) - row["value"])**2 for row in records)
        self.pinning = float(minimize_scalar(error, bounds=(0.85, 1.15), method="bounded").x)
        return self

    def predict(self, experiments):
        return np.array([signal(q, self.pinning) for q in experiments])
