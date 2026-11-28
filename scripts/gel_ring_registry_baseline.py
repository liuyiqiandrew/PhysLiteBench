import numpy as np
from registry import Printer
from gel import joint


class Model(Printer):
    @staticmethod
    def pair_weights(f0, f1, f2, u, v):
        # Each full daughter has its exact one-ring marginal, but receives a separate draw of the common trunk.
        one = np.ones(4)
        return joint(f0, f1, one, u, u) * joint(f0, f2, one, v, v)
