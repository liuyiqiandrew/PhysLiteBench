"""Each dye has the right individual wall-conditioned law, on independent chains."""
from camera import Camera
from passage import density


class Model(Camera):
    @staticmethod
    def response(z, w, walls, gate):
        total, _, second, _ = density(z,w,walls)
        return second/total
