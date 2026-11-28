from apparatus import Calibrated
from swap import values


class Model(Calibrated):
    def read(self, q):
        return values(q, self.activity)[q["observable"]]
