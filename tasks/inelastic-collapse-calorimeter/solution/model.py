from bead import Meter
from renewal import signal


class Model(Meter):
    def read(self, q):
        return signal(q, self.diffusivity)
