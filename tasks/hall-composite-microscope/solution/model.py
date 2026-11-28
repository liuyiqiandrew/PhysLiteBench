from meter import Meter
from film import signal
from thermal import voltage
from calorimeter import heat


class Model(Meter):
    def read(self, q):
        if q["mode"] == "heat":
            return heat(q, self.gain)
        return voltage(q, self.gain) if q["mode"] == "noise" else signal(q, self.gain)
