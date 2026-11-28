from apparatus import Calibrated
from spectral import reading


class Model(Calibrated):
    def read(self, q):
        return reading(q, self.diffusivity)
