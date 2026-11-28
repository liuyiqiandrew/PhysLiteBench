"""Preserve the exact parity of each component but decohere different Slater determinants."""
from edge import reading
from apparatus import Readout


class Model(Readout):
    def fringe(self, q):
        return reading(q, dephase=True)
