import math
import numpy as np


class Detector:
    def fit(self, records):
        self.activity = float(
            np.mean(
                [math.sqrt(r["value"] / (1 - r["value"])) / r["input"]["ratio"] for r in records]
            )
        )
        return self

    def predict(self, inputs):
        values = []
        for q in inputs:
            if q["mode"] == "bond":
                weight = (self.activity * q["ratio"]) ** 2
                values.append(weight / (1 + weight))
            else:
                values.append(self.loop(q))
        return np.asarray(values, dtype=float)
