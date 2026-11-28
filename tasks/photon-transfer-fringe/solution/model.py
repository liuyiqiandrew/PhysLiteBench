from optics import Calibrated, calibration_signal
from waveguide import waveguide


class Model(Calibrated):
    def reading(self, q):
        return waveguide(q)[0] if "rates" in q else calibration_signal(q)
