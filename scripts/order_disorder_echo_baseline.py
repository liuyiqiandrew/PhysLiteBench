from blocks import response
from instrument import Instrument


class Model(Instrument):
    """Treat the transverse dual interval as an ordinary order dipole."""

    def amplitude(self, q):
        return response(q, ignore_disorder=True)
