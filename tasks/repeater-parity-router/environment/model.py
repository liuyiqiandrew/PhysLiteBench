"""Serve the two receivers from independent, parity-selected loadings."""
from apparatus import Calibrated, single


class Model(Calibrated):
    def read(self, q):
        if q["observable"] == "contention":
            return 0.0
        p, x = single(q, self.activity, 0)
        b, y = single(q, self.activity, 1)
        return p * b if q["observable"] == "delivery" else x * y
