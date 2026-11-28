from camera import Camera
from passage import response


class Model(Camera):
    response = staticmethod(response)
