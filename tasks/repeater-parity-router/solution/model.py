from apparatus import Calibrated
from tagged import value


class Model(Calibrated):
    def read(self, q):
        return value(q, self.activity)
