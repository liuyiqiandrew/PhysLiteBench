"""Count eligible terminal tracks uniformly instead of resolving the last microscopic bond."""
from camera import Camera
from waves import signal


class Model(Camera):
    def measure(self,q):
        return signal(q,self.rate,coarse=True)
