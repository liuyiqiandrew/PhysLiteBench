import numpy as np


def factors(setting, k, single, exponent):
    """Return even and odd parity propagators for one momentum block."""
    h, hop, pair, phase = setting
    xi = h - 2 * hop * np.cos(k)
    if single:
        return np.array([[np.exp(-exponent * xi / 2)]]), np.exp(exponent * xi / 2)
    z = 2j * pair * np.exp(1j * phase) * np.sin(k)
    matrix = np.array([[-xi, z.conjugate()], [z, xi]])
    energy = np.hypot(xi, abs(z))
    if energy == 0:
        return np.eye(2), 2
    even = np.cosh(exponent * energy) * np.eye(2)
    even += np.sinh(exponent * energy) / energy * matrix
    return even, 2


def periodic_trace(expt):
    n = expt["n"]
    total_time = sum(dt for dt, _ in expt["pulses"])
    amplitude = weight_product = 1.0
    for m in range(n):
        k = 2 * np.pi * m / n
        if k > np.pi + 1e-10:
            break
        single = abs(np.sin(k)) < 1e-10
        thermal, odd_weight = factors(expt["prep"], k, single, complex(-expt["beta"]))
        weight = thermal.trace().real
        evolution, odd_amp = factors(
            expt["prep"], k, single, -expt["beta"] + 1j * total_time
        )
        for dt, setting in expt["pulses"]:
            pulse, odd_pulse = factors(setting, k, single, -1j * dt)
            evolution = pulse @ evolution
            if single:
                odd_amp *= odd_pulse
        amplitude *= evolution.trace() + odd_amp
        weight_product *= weight + odd_weight.real
    return amplitude, weight_product


def fringe(expt):
    amplitude, weight = periodic_trace(expt)
    return (np.exp(-1j * expt["angle"]) * amplitude / weight).real


class Model:
    def fit(self, records):
        x = np.array([fringe(r["input"]) / r["sigma"] for r in records])
        y = np.array([r["value"] / r["sigma"] for r in records])
        self.visibility = np.clip((x @ y) / (x @ x), .65, .95)
        return self

    def predict(self, inputs):
        return self.visibility * np.array([fringe(e) for e in inputs])
