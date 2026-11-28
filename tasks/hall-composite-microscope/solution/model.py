from meter import Meter
from film import signal
from noise import voltage


class Model(Meter):
    def read(self, q):
        return voltage(q, self.gain) if q["mode"] == "noise" else signal(q, self.gain)
