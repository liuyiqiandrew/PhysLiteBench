from registry import Printer
from gel import joint


class Model(Printer):
    pair_weights = staticmethod(joint)
