from meter import Meter
from duality import signal


class Model(Meter):
    def read(self, q):
        return signal(q, self.gain)
