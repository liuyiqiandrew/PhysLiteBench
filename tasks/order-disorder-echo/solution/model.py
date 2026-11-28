from blocks import response
from instrument import Instrument


class Model(Instrument):
    def amplitude(self, q):
        return response(q)
