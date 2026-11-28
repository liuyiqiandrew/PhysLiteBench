import numpy as np
from scipy.optimize import minimize_scalar


def right(point):
    x, y = point.real, point.imag
    return (1.0 + x / np.hypot(x, y)) / 2.0


class Camera:
    @staticmethod
    def response(z, w, walls, gate):
        return right(w)

    def predict(self, inputs):
        readings = []
        for r in inputs:
            z, w = [complex(x, self.aspect * y) for x, y in r["dyes"]]
            if r["gate"] == "off":
                readings.append(right(w))
            else:
                readings.append(self.response(z, w, r["walls"], r["gate"]))
        return np.array(readings, dtype=float)

    def fit(self, rows):
        def residual(aspect):
            self.aspect = aspect
            return sum((self.predict([r["input"]])[0] - r["value"]) ** 2 for r in rows)
        self.aspect = minimize_scalar(residual, bounds=(0.78, 1.22), method="bounded",
                                      options={"xatol": 1e-13}).x
        return self
