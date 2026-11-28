"""Ideal voltage clamp after every carrier birth, preserving all timing selections."""
from current import Calibrated
from geometry import reading


class Model(Calibrated):
    def waveform(self, q):
        return reading(dict(q, feedback=0.0), self.activity, 18)[0]
