import numpy as np


def response(e):
    r, z = float(e['drive_ratio']), float(e['relative_heat'])
    r2 = r*r
    q = 2*r2*z

    # Legendre transform of the bulk heat generating function.  The
    # unconstrained saddle is valid only while both free endpoint integrals
    # converge.  At their boundaries the transform continues linearly.
    if z <= -1.0:
        rate = -q
    elif r2 < 1.0/3.0 and z >= (3.0-r2)/(1.0-3.0*r2):
        boundary = (1.0-r2)/(1.0+r2)
        edge_root = (1.0-3.0*r2)/(1.0+r2)
        rate = boundary*q - 1.0 + edge_root
    else:
        rate = np.sqrt((1.0+r2)*(1.0+r2*z*z)) - r2*z - 1.0
    return float(e['stiffness'])*rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        unit_rates = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        weights = 1.0 / np.square(sigmas)
        self.mobility = float(np.clip(np.sum(weights*unit_rates*values) /
                                      np.sum(weights*unit_rates*unit_rates),
                                      0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
