"""Add the stroke gyro moment but discard the rigid motion of the zero-area cycle."""
from apparatus import Camera
from passive import fringe
from gradient import pickup


class Model(Camera):
    def fringe(self, q):
        return pickup(q, self.mobility, body_drive=False) if "pickup" in q else fringe(q, self.mobility)
