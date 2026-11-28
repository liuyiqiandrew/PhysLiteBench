"""Measure the intermediate atomic excitation and discard the measurement record."""
from optics import Calibrated, calibration_signal
from interferometer import delay_signal


class Model(Calibrated):
    def reading(self, q):
        return (
            delay_signal(q, measure=True)[0]
            if "temporal" in q
            else calibration_signal(q)
        )
