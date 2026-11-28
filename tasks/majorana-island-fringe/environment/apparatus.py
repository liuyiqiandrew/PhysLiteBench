import numpy as np


def contacts(q):
    d = 2 * q["n"]
    if q["frame"] == "site":
        return np.eye(d)
    j = np.arange(d) + .5
    matrix = np.sqrt(2 / d) * np.cos(np.pi * np.arange(d)[:, None] * j / d)
    matrix[0] /= np.sqrt(2)
    return matrix


def polarizations(q):
    j = np.arange(q["n"]) + .5
    energies = 1 + q["ripple"] * np.cos(2 * np.pi * j / q["n"])
    return np.tanh(.5 * q["beta"] * energies)


class Readout:
    def fit(self, records):
        signal = np.array([np.cos(r["input"]["phase"]) for r in records])
        measured = np.array([r["value"] for r in records])
        self.visibility = float(np.clip(signal @ measured / (signal @ signal), .65, .95))
        return self

    def predict(self, inputs):
        return self.visibility * np.array([self.fringe(q) for q in inputs], dtype=float)
