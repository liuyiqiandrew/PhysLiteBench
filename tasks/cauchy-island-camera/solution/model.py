from apparatus import Calibrated
from arch import reading


class Model(Calibrated):
    def read(self, q):
        return reading(q, self.diffusivity)[
            ("final", "history", "duration").index(q["observable"])
        ]
