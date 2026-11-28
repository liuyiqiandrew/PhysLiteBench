"""Measure both charge coordinates on one avalanche; trigger the third read on an independent avalanche."""
from current import Calibrated
from geometry import values


class Model(Calibrated):
    def waveform(self, q):
        intensity, wave, base = values(q, self.activity, 10)
        return base if q["observable"] == "waveform" else intensity
