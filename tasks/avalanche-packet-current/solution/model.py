from current import Calibrated
from geometry import values


class Model(Calibrated):
    def waveform(self, q):
        intensity, wave, base = values(q, self.activity, 10)
        return wave if q["observable"] == "waveform" else intensity
