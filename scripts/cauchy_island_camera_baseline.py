"""The actual takeoff coordinate and both height gates are retained, but time is sampled from its unselected arch marginal."""
from apparatus import Calibrated
from arch import reading


class Model(Calibrated):
    def read(self, q):
        return reading(q, self.diffusivity, uncoupled=True)[
            ("final", "history", "duration").index(q["observable"])
        ]
