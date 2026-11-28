from pump import PumpSurvey
from spectral import spectral_signal


class Model(PumpSurvey):
    def reading(self, q):
        return spectral_signal(q)[0] if "spectral" in q else super().reading(q)
