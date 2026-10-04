import numpy as np
from scipy.optimize import minimize_scalar

RATIOS = np.array([1., 1.7, 2.6])


def polarizability(strength, orientation):
    values = strength*RATIOS
    return orientation@np.diag(values/(1-1j*values/(6*np.pi)))@orientation.T


def response(experiment, strength):
    orientation = np.array(experiment['orientation'])
    field = np.array(experiment['field_real'])+1j*np.array(experiment['field_imag'])
    dipole = polarizability(strength, orientation)@field
    if experiment['observable'] == 'force':
        vector = .5*np.imag(np.vdot(field, dipole))*np.array(experiment['direction'])
    else:
        vector = .5*np.real(np.cross(dipole, field.conj()))

    return float(np.array(experiment['axis'])@vector)


def predict_at(experiments, strength):
    return np.array([response(e, strength) for e in experiments])


class Model:
    def __init__(self):
        self.response_strength = None

    def fit(self, records):
        design = []
        for record in records:
            e = record['input']
            field = np.array(e['field_real'])+1j*np.array(e['field_imag'])
            body = np.array(e['orientation']).T@field
            design.append(.5*np.dot(e['axis'], e['direction'])*abs(body)**2)
        design = np.array(design)
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def objective(strength):
            raw = strength*RATIOS
            alpha = raw/(1-1j*raw/(6*np.pi))
            residual = (design@alpha.imag-values)/sigma
            return float(residual@residual)
        self.response_strength = float(minimize_scalar(objective, bounds=(.6, 1.6), method='bounded', options={'xatol':1e-13}).x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.response_strength)
