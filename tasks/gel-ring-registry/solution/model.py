from registry import Printer
from gel import tightened


class Model(Printer):
    dynamics = staticmethod(tightened)
