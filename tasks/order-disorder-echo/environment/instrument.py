import numpy as np


class Instrument:
    def fit(self, records):
        phase = np.array([r["input"]["phase"] for r in records])
        c = np.cos(phase)
        self.visibility = float(c @ np.array([r["value"] for r in records]) / (c @ c))
        return self

    def predict(self, inputs):
        return np.array(
            [
                self.visibility
                * (
                    np.cos(q["phase"])
                    if "pulses" not in q
                    else np.real(self.amplitude(q) * np.exp(-1j * q["phase"]))
                )
                for q in inputs
            ],
            dtype=float,
        )
