import numpy as np


class Readout:
    def fit(self, records):
        e = np.array([r["input"]["threshold"] for r in records])
        attenuation = -np.log([r["value"] for r in records])
        self.depth_scale = float(np.clip((e @ e) / (e @ attenuation), 0.9, 1.1))
        return self

    def predict(self, inputs):
        return np.asarray(
            [
                np.exp(-q["threshold"] / self.depth_scale)
                if "threshold" in q
                else self.memory(q)
                for q in inputs
            ],
            dtype=float,
        )
