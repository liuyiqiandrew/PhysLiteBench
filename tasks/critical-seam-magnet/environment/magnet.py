import numpy as np


class Magnet:
    def fit(self, records):
        h = np.array([r["input"]["bias"] for r in records])
        observed = np.arctanh([r["value"] for r in records])
        self.gain = float(np.clip((h @ observed) / (h @ h), 0.75, 1.15))
        return self

    def predict(self, inputs):
        return np.asarray(
            [
                np.tanh(self.gain * q["bias"]) if "sites" not in q else self.measure(q)
                for q in inputs
            ],
            dtype=float,
        )
