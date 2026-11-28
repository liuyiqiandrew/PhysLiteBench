"""Replace a jointly postselected four-spin mean by the product of its two marginal means."""
from apparatus import Calibrated
from swap import values


class Model(Calibrated):
    def read(self, q):
        return values(q, self.activity, shortcut="independent_outputs")[q["observable"]]
