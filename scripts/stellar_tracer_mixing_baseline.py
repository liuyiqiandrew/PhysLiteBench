"""Completed control: reversible adiabatic evolution for every field change."""
import numpy as np


def factors(q):
    m, b = q["steps"][-1] if q["steps"] else q["slow"]
    rp, ra = q["peri"], q["apo"]
    sp, sa = np.hypot(b, [rp, ra])
    energy = -m / (sa + sp)
    angular = rp * ra * np.sqrt(-2 * energy / ((sp + b) * (sa + b)))
    jr = m / np.sqrt(-2 * energy) - (angular + np.sqrt(angular**2 + 4*m*b)) / 2
    initial_binding = 0.5 / (jr + (angular + np.sqrt(angular**2 + 4)) / 2)**2
    period = 2*np.pi*m / (-2*energy)**1.5
    return 8*np.pi**2 * angular * period, initial_binding / 0.4


class Model:
    def fit(self, records):
        x, y = [], []
        for row in records:
            volume, binding = factors(row["input"])
            x.append(np.log(binding))
            y.append(np.log(row["value"] / volume))
        self.exponent = float(np.dot(x, y) / np.dot(x, x))
        return self

    def predict(self, experiments):
        return np.array([volume * binding**self.exponent
                         for volume, binding in map(factors, experiments)])
