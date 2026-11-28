"""A Gaussian spectrometer with fixed electronic noise in induced-charge units."""
import numpy as np
from current import Calibrated
from transport import packet_law


class Model(Calibrated):
    def waveform(self, q):
        sigma = 0.04
        spec = q["spectrum"]
        selected = tuple(slice(None) if c is None else c for c in q["multiplicities"])
        raw, _ = packet_law(q, self.activity, 8)
        noisy = dict(q, tag=[*spec["window"], spec["charge"], sigma])
        mass, moment = packet_law(noisy, self.activity, 8)
        m = mass[selected].sum()
        if q["observable"] == "waveform":
            return moment[selected].sum() / m
        acceptance = m / raw[selected].sum()
        return (
            acceptance
            if spec["part"] == "line"
            else acceptance / (np.sqrt(2 * np.pi) * sigma)
        )
