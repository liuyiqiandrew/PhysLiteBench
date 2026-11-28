from current import Calibrated
from geometry import reading


class Model(Calibrated):
    def waveform(self, q):
        return reading(q, self.activity, 18)[0]
