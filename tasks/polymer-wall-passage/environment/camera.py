import numpy as np
from scipy.optimize import minimize_scalar


def single_at_reach(radius, z):
    return .8*z.imag**2*(1/abs(z)**2-1/radius**2)


class Camera:
    @staticmethod
    def response(radius, z, w, walls):
        return single_at_reach(radius,w)

    def predict(self, inputs):
        values = []
        for r in inputs:
            z,w = [complex(x,self.aspect*y) for x,y in r["dyes"]]
            if r["gate"] == "off":
                values.append(single_at_reach(r["radius"],w))
            else:
                values.append(self.response(r["radius"],z,w,r["walls"]))
        return np.array(values, dtype=float)

    def fit(self, rows):
        def objective(aspect):
            self.aspect = aspect
            return sum((self.predict([r["input"]])[0]-r["value"])**2 for r in rows)
        self.aspect = minimize_scalar(objective, bounds=(.78,1.22), method="bounded",
                                      options={"xatol":1e-13}).x
        return self
