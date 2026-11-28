"""Trigger a separate avalanche for the waveform read, keeping the same terminal selections."""
from current import Calibrated
from geometry import reading


class Model(Calibrated):
    def waveform(self, q):
        return reading(q, self.activity, independent_read=True)
