"""Both daughters recoil to the same randomly chosen side of the parent, with correct one-daughter marginals."""
from apparatus import Calibrated
from tree import solve


class Model(Calibrated):
    def read(self, q):
        return solve(q, self.diffusivity, 88, shortcut="equal recoil")[
            ("red", "coincidence", "clean").index(q["observable"])
        ]
