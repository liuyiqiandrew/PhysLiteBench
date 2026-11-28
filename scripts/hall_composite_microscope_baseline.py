"""Exact bulk and phase means, with the two local probes sampled independently inside a phase."""
from meter import Meter
from duality import signal


class Model(Meter):
    def read(self, q):
        return signal(q, self.gain, independent=True)
