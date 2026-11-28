"""Measure the spin storage number before reading the occupied cells."""
from optics import Calibrated, calibration_signal
from returning import echo


class Model(Calibrated):
    def reading(self, q):
        return echo(q, measured=True)[0] if "store" in q else calibration_signal(q)
