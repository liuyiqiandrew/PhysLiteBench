import numpy as np

from scipy.integrate import quad

def predict(experiments, temperature):
    s = np.sqrt(1.380649e-23*temperature/(39.948*1.66053906660e-27))
    out = []
    for e in experiments:
        z = e['threshold']/s
        if e['component'] in ('x', 'y'):
            out.append(quad(lambda u: np.exp(-u*u/2)/np.sqrt(2*np.pi), -np.inf, z)[0])
        else:
            out.append(quad(lambda u: u*np.exp(-u*u/2), 0., max(0., z))[0])
    return np.array(out)


def calibration_inputs():
    return [dict(component=c, threshold=float(v)) for c in ['x', 'y'] for v in np.linspace(-600, 600, 50)]

def hidden_inputs():
    return {f'axial_{j}': [dict(component='z', threshold=float(v)) for v in np.linspace(lo, hi, 24)]
            for j, (lo, hi) in enumerate([(30, 240), (250, 480), (490, 800)])}

PARAMETER = 'temperature'
TRUE_PARAMETER = 430.0
BOUNDS = (150.0, 1000.0)
