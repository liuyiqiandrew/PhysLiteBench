"""Use exact herald routing but draw a new phase for every elementary emission."""
from apparatus import Calibrated
from flow import weights
from fringe import ensemble


class Model(Calibrated):
    def read(self, q):
        if q["observable"] == "fringe":
            return ensemble(q, self.activity, fresh=True)[0]
        joint, normalization = weights(q, self.activity)
        if q["observable"] == "delivery":
            return joint / normalization
        separate, _ = weights(q, self.activity, duplicate=True)
        return 1 - joint / separate
