"""Independent cooldown at readout, with exact discrete-defect spectra at preparation."""
from apparatus import Readout
from echo import reading


class Model(Readout):
    def fringe(self, q):
        return reading(q,reset=True)
