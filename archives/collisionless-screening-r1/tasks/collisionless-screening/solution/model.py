import numpy as np
from scipy.optimize import minimize_scalar


def velocity_response(zeta):
    z = np.asarray(zeta, dtype=float)
    return 15/4 * (2*z*z - 4/3 + (z**3-z)*np.log(np.abs((z-1)/(z+1))))


def predict_at(experiments, density):
    k = np.asarray([e["wavenumber"] for e in experiments])
    width = np.asarray([e["velocity_width"] for e in experiments])
    omega = np.asarray([e["frequency"] for e in experiments])
    z = omega/(k*width)
    h = velocity_response(z)
    h = h.astype(complex) + 1j*np.pi*15/4*(z**3-z)*(np.abs(z)<1)
    return np.real(1/(1-density*h/(k*width)**2))


class Model:
    def __init__(self):
        self.density = None

    def fit(self, records):
        experiments = [r["input"] for r in records]
        observed = np.asarray([r["value"] for r in records])
        sigma = np.asarray([r["sigma"] for r in records])
        def loss(density):
            residual = (predict_at(experiments, density)-observed)/sigma
            return float(residual @ residual)
        result = minimize_scalar(loss, bounds=(.6,1.4), method="bounded",
                                 options={"xatol":1e-12})
        candidates = [.6, float(result.x), 1.4]
        self.density = min(candidates, key=loss)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.density)
