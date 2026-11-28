from apparatus import Camera
from monitor import monitored
from passive import fringe


class Model(Camera):
    def fringe(self, q):
        return monitored(q, self.mobility) if "monitor" in q else fringe(q, self.mobility)
