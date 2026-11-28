from camera import Camera
from passage import cumulative


class Model(Camera):
    @staticmethod
    def response(radius,z,w,walls):
        one,both=cumulative(radius,z,w,walls)
        return both/one
