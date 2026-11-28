"""Use the applied-field illumination of each port to collect its local noise."""
from meter import Meter
from film import signal
from noise import voltage


class Model(Meter):
    def read(self, q):
        return voltage(q, self.gain, adjoint=False) if q["mode"] == "noise" else signal(q, self.gain)
