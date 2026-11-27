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


def rotary(q, pinning):
    h, ratio = q["amplitude"], q["cutting"]
    sense = q["sense"]
    start = 1e-7
    mobility = ratio**2
    pitch = np.sqrt(2 * mobility)
    scale = np.sqrt(1 + (pitch/ratio)**2) / pinning
    k0 = scale * start**2 / (2 + 1j*pitch)

    def slope(b, y):
        re, im, _, _ = y
        tan = -mobility * re / im
        dx = (1 + b) * np.hypot(1, tan / ratio) / pinning
        dangle = tan / b
        return [b * dx + im * dangle, -re * dangle, dangle, dx]

    sol = solve_ivp(slope, (start, h), [k0.real, k0.imag, 0, scale*start],
                    method="DOP853", rtol=1e-9, atol=[1e-22, 1e-22, 1e-9, 1e-14],
                    dense_output=True)
    _, _, angle, depth = sol.y[:, -1]
    x = q["depth"]
    if x >= depth - scale*start:
        return 0.
    b = h if x == 0 else brentq(lambda b: sol.sol(b)[3] - depth + x, start, h)
    re, im, local, _ = sol.sol(b)
    orientation = np.exp(1j*sense*(local-angle))
    if q["readout"] == "induction":
        value = b * orientation
    else:
        value = (sense*re + 1j*im) * orientation
    return value.real if q["axis"] == "y" else value.imag


def signal(q, pinning):
    if q["readout"] in ("induction", "electric"):
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
