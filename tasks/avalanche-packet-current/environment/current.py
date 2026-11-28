"""Dark-flight calibration and charge induced by straight carrier tracks."""
import numpy as np


def induced(t, pairs, velocities):
    ve, vh = velocities
    return sum(
        min(ve * max(0, t - t0), 1 - x) + min(vh * max(0, t - t0), x) for t0, x in pairs
    )


class Calibrated:
    def fit(self, rows):
        lengths = np.array([row["input"]["active_length"] for row in rows])
        logp = np.log([row["value"] for row in rows])
        self.activity = float(-(lengths @ logp) / (lengths @ lengths))
        return self

    def predict(self, inputs):
        return np.array(
            [
                np.exp(-self.activity * q["active_length"])
                if "active_length" in q
                else self.waveform(q)
                for q in inputs
            ],
            dtype=float,
        )
