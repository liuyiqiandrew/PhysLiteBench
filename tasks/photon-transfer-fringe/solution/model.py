from optics import Calibrated, calibration_signal
from returning import echo


class Model(Calibrated):
    def reading(self, q):
        return echo(q)[0] if "store" in q else calibration_signal(q)
