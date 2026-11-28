from apparatus import Calibrated
from tree import solve


class Model(Calibrated):
    def read(self, q):
        return solve(q, self.diffusivity, 88)[
            ("red", "coincidence", "clean").index(q["observable"])
        ]
