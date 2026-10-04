import numpy as np


def velocity_response(zeta):
    z = np.asarray(zeta, dtype=float)
    return 15/4 * (2*z*z - 4/3 + (z**3-z)*np.log(np.abs((z-1)/(z+1))))


def predict_at(experiments, density):
    k = np.asarray([e["wavenumber"] for e in experiments])
    width = np.asarray([e["velocity_width"] for e in experiments])
    omega = np.asarray([e["frequency"] for e in experiments])
    z = omega/(k*width)
    h = velocity_response(z)
    return np.real(1/(1-density*h/(k*width)**2))


class Model:
    def __init__(self):
        self.density = None

    def fit(self, records):
        raise NotImplementedError("Fit density from the supplied calibration records.")

    def predict(self, experiments):
        return predict_at(experiments, self.density)
