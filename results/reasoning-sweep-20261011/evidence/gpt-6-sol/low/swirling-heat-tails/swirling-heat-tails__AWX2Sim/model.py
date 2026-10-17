import numpy as np


def response(e):
    """Stationary heat rate per unit mobility.

    The work SCGF is 1-sqrt(1+4*r**2*lambda*(1-lambda)).
    The unconditioned endpoint energies restrict its heat transform to
    lambda <= 1 and, for weak drives, a finite lower endpoint.
    """
    r, z = float(e['drive_ratio']), float(e['relative_heat'])
    r2 = r*r
    saddle = (1-z*np.sqrt((1+r2)/(1+r2*z*z)))/2
    lower = (r2-1)/(1+r2) if r2 <= 1/3 else (1-np.sqrt(1+1/r2))/2
    lam = np.clip(saddle, lower, 1.0)
    root = np.sqrt(max(0.0, 1+4*r2*lam*(1-lam)))
    return e['stiffness']*(root-1-2*r2*z*lam)


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        basis = np.asarray([response(record['input']) for record in records])
        values = np.asarray([record['value'] for record in records])
        weights = 1/np.square([record['sigma'] for record in records])
        self.mobility = float(np.clip(np.sum(weights*basis*values) /
                                      np.sum(weights*basis*basis), 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
