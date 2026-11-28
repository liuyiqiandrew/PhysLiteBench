"""Collect a separately normalized sample from each eligible fabrication domain."""
from camera import Camera
from waves import signal


class Model(Camera):
    def measure(self,q):
        return signal(q,self.rate,reprepare=True)
