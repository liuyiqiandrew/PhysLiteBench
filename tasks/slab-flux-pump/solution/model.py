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


def material(b, re, im, n, cut, j):
    size = np.hypot(b, n)
    jc = j / (1 + size)
    # Parameterize the tilted transverse circle by its in-plane ordinate t.
    # A plane at fixed normal current intersects the two cutting facets.
    a = re * size / n
    optimum = jc * a / np.hypot(a, im)
    limit = cut * jc * abs(n) / b
    t = np.clip(optimum, -limit, limit)
    j1 = size / n * t
    j2 = np.copysign(np.sqrt(max(0.0, (jc - t) * (jc + t))), im if im else -1.0)
    if abs(optimum) <= limit:
        ex = -b / n * re
        # A direct ratio also works at im=0 and gives the smooth ellipse tip.
        j2 = jc * im / np.hypot(a, im)
    else:
        ex = n / b * (re - im / j2 * j1)
    return j1, j2, ex


@lru_cache(maxsize=128)
def film_profile(h, n, cut, coat, g, hall, s, kappa, j):
    def rhs(log_u, state):
        m, re, im, theta, w = state
        u = np.exp(log_u)
        j1, j2, ex = material(u * m, re, im, n, cut, j)
        twist = j1 / m
        return [-j2 - m, m - 2 * re + twist * im, -2 * im - twist * re, twist, u**3 * ex]

    def solve(u, v, state):
        return solve_ivp(
            rhs,
            np.log([u, v]),
            state,
            method="DOP853",
            rtol=2e-9,
            atol=8e-13,
            dense_output=True,
        ).sol

    start = 1.0e-7
    m = j / (1 + abs(n)) / np.sqrt(3.0)
    k = m / (2 + 1j * np.sqrt(2.0))
    substrate = solve(start, 1.1, [m, k.real, k.imag, 0.0, 0.0])

    def active(u):
        m, re, im, theta, w = substrate(np.log(u))
        e = u * u * (re + 1j * im)
        b = u * m + g * (1j - s * hall * np.sign(n)) * e + 1j * kappa * e / abs(e)
        delta = np.angle(b)
        k = (re + 1j * im) * np.exp(-1j * delta)
        return solve(u, u + coat, [abs(b) / u, k.real, k.imag, theta + delta, w])

    def excluded(b0):
        z = 2.0e-8
        # Here u=1+z is an arbitrary shifted coordinate in the coating.
        j1, j2, _ = material(b0, 1.0, 0.0, n, cut, j)
        slope = -j2 + 1j * j1
        b = b0 + z * slope
        e = z * (b0 + z * slope / 2.0)
        u = 1 + z
        k = e * (b / abs(b)).conjugate() / u**2
        return solve(u, 1 + coat, [abs(b) / u, k.real, k.imag, np.angle(b), 0.0])

    def amp(path, u):
        return u * path(np.log(u))[0]

    limit_amplitude = amp(excluded(kappa), 1 + coat)
    if limit_amplitude >= h - 5e-7:
        b0 = (
            kappa
            if limit_amplitude <= h
            else brentq(lambda b: amp(excluded(b), 1 + coat) - h, 1.0e-4, kappa)
        )
        outer = excluded(b0)
        offset = 1.0
        front = coat
        power = 0.0
    else:

        def residual(u):
            return amp(active(u), u + coat) - h

        offset = brentq(residual, start, 1.0)
        outer = active(offset)
        front = offset + coat
        e = np.linalg.norm(substrate(np.log(offset))[1:3]) * offset**2
        power = g * e**2 + kappa * e
    high = offset + coat
    surface = outer(np.log(high))
    return front, high, substrate, outer, surface, power


def film(q, pinning):
    coat, s = q["coating"], q["sense"]
    front, high, inner, outer, surface, power = film_profile(
        q["amplitude"],
        q["normal"],
        q["cutting"],
        coat,
        q["conductance"],
        q["hall"],
        s,
        q["film"],
        pinning,
    )
    if q["readout"] == "sheet_power":
        return power
    x = q["depth"]
    # For a shielded film, the limiting outer b at x == coating is finite.
    if x > coat and front - x <= 1.0e-7:
        return s * surface[4] if q["readout"] == "normal_voltage" else 0.0
    u = high - x if x <= coat else front - x
    state = (outer if x <= coat else inner)(np.log(u))
    if q["readout"] == "normal_voltage":
        return s * (surface[4] - state[4])
    orient = np.exp(1j * s * (state[3] - surface[3]))
    value = orient * (
        u * state[0] if q["readout"] == "induction" else u * u * (s * state[1] + 1j * state[2])
    )
    return value.real if q["axis"] == "y" else value.imag


def signal(q, pinning):
    if "film" in q:
        return film(q, pinning)
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
