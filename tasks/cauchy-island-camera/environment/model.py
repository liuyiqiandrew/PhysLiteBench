"""At most one fission per initially live carrier, located at its initial position."""
import numpy as np
from scipy.signal import convolve2d
from apparatus import Calibrated, terminal


def stops(q, x):
    red, blue = terminal(q, x)
    return np.array([[1 - red - blue, blue], [red, 0.0]])


class Model(Calibrated):
    def read(self, q):
        d = q["recoil"]

        def family(x):
            single = stops(q, x)
            if abs(x) >= 1.0 or q["rate"] == 0.0:
                return single
            probability = 1.0 - np.exp(
                -q["rate"] * np.sqrt(1.0 - x * x) / self.diffusivity
            )
            pair = convolve2d(stops(q, x - d), stops(q, x + d))
            return (1.0 - probability) * np.pad(
                single, ((0, 1), (0, 1))
            ) + probability * pair

        x = q["launch"]
        counts = (
            family(x)
            if q["source"] == "single"
            else convolve2d(family(x - d), family(x + d))
        )
        red = counts[1:, :].sum()
        both = counts[1:, 1:].sum()
        if q["observable"] == "red":
            return red
        if q["observable"] == "coincidence":
            return both / red
        return counts[1, 1] / both
