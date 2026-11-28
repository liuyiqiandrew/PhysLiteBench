"""Interface and uniform-phase Hall calibration."""
import numpy as np


def homogeneous(q, gain):
    e = np.asarray(q["drive"]) @ np.array([1.0, 1j])
    m = q["material"]
    j = (np.asarray(m["longitudinal"]) + 1j * gain * q["bias"] * np.asarray(m["hall"])) @ e
    return float((np.exp(-1j * q["angle"]) * (np.asarray(q["contact"]) @ j)).real)


class Meter:
    def fit(self, records):
        inputs = [r["input"] for r in records]
        base = np.array([homogeneous(q, 0.0) for q in inputs])
        slopes = np.array([homogeneous(q, 1.0) for q in inputs]) - base
        self.gain = float(
            np.clip(
                slopes @ (np.array([r["value"] for r in records]) - base) / (slopes @ slopes), 0.65, 1.15
            )
        )
        return self

    def predict(self, inputs):
        return np.array(
            [homogeneous(q, self.gain) if q["mode"] == "calibration" else self.read(q) for q in inputs],
            dtype=float,
        )
