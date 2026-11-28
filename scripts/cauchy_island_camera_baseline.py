"""A second independent trajectory supplies the clock at the same launch and geometry."""
from apparatus import Calibrated
from spectral import reading


class Model(Calibrated):
    def read(self, q):
        return reading(q, self.diffusivity, detached=True)
