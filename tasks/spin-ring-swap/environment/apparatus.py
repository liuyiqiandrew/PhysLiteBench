"""Calibrate the optical visibility on the blank phase records."""
import numpy as np


class Readout:
    def __init__(self):
        self.visibility = 0.8

    def fit(self, records):
        x = np.cos([r["input"]["phase"] for r in records])
        y = [r["value"] for r in records]
        self.visibility = float(np.clip(x @ y / (x @ x), 0.65, 0.95))
        return self

    def predict(self, inputs):
        return self.visibility * np.array([self.fringe(x) for x in inputs], dtype=float)
