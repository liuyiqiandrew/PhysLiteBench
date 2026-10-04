import numpy as np

RHO0 = 1.
BULK = 1.
EXPONENT = 5.


def fields(length, mode, frequency_ratio, drive, viscosity):
    wavevector = mode*np.pi/length
    frequency = frequency_ratio*np.sqrt(BULK/RHO0)*wavevector
    velocity = drive/(viscosity*wavevector**2+1j*(BULK*wavevector**2/frequency-RHO0*frequency))
    density = -1j*RHO0*wavevector*velocity/frequency
    return velocity, density


def predict_at(experiments, viscosity):
    result = []
    for e in experiments:
        velocity, density = fields(e['length'], e['mode'], e['frequency_ratio'], e['drive'], viscosity)
        if e['readout'] == 'density':
            value = abs(density)
        else:
            curvature = (EXPONENT-1)*BULK/RHO0**2
            value = .25*RHO0*abs(velocity)**2+.125*curvature*abs(density)**2
        result.append(float(value))
    return np.array(result)


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        from scipy.optimize import minimize_scalar
        records = list(records)
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        objective = lambda eta: float(np.sum(((predict_at(experiments, eta)-values)/sigma)**2))
        optimum = minimize_scalar(objective, bounds=(.08, .18), method='bounded', options={'xatol':1e-13})
        self.viscosity = float(optimum.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
