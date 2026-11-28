"""Use the correct pilot-selected inventories but an independent source phase for the remote readout."""
from apparatus import Calibrated
from tagged import value


class Model(Calibrated):
    def read(self, q):
        return value(q, self.activity, detach=True)
