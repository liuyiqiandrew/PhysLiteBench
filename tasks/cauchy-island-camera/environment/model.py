"""One arch launched from the source, with separate height and duration marginals."""
from apparatus import Calibrated, arch_below, arch_time_cdf


class Model(Calibrated):
    def read(self, q):
        displacement = q["landing"] - q["launch"]
        if q["observable"] == "final":
            lo, hi = q["heights"]
            return arch_below(displacement, hi) - arch_below(displacement, lo)
        if q["observable"] == "history":
            return 1.0
        lo, hi = q["window"]
        return arch_time_cdf(displacement, hi, self.diffusivity) - arch_time_cdf(
            displacement, lo, self.diffusivity
        )
