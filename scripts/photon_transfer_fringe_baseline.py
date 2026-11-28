"""Record which converter port emitted the trigger photon."""
from optics import Calibrated, calibration_signal
from spectral import spectral_signal
from herald import conversion_signal


class Model(Calibrated):
    def reading(self, q):
        if "conversion" in q:
            return conversion_signal(q, separate=True)[0]
        if "spectral" in q:
            return spectral_signal(q)[0]
        return calibration_signal(q)
