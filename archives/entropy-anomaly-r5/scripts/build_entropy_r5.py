"""One-time author construction of the entropy r5 staged candidate."""
from pathlib import Path
import json

STAGE = Path(__file__).resolve().parents[1]
TASK = STAGE / 'tasks/entropy-anomaly'
REPO = STAGE.parents[1]

common = '''from functools import lru_cache
import numpy as np


@lru_cache(None)
def spatial_grid(points):
    x = np.arange(points)*2*np.pi/points
    modes = np.fft.fftfreq(points, 1/points)
    derivative = np.fft.ifft(1j*modes[:, None]*np.fft.fft(np.eye(points), axis=0), axis=0).real
    return x, derivative


def positional_state(experiment, friction, points=129):
    x, derivative = spatial_grid(points)
    derivative = derivative*experiment['wavenumber']
    base = experiment['temperature']
    contrast = experiment['contrast']
    bath_contrast = experiment['bath_contrast']
    temperatures = np.array([base*(1+sign*bath_contrast)*(1+contrast*np.cos(x)) for sign in [1, -1]])
    gradients = np.array([-base*(1+sign*bath_contrast)*contrast*experiment['wavenumber']*np.sin(x) for sign in [1, -1]])
    gamma_y = friction*experiment['drag_ratio']
    magnetic = experiment['magnetic']
    effective_force = experiment['force_x']+magnetic*experiment['force_y']/gamma_y
    effective_drag = friction+magnetic**2/gamma_y
    switch_rate = experiment['switch_rate']
    operators = [-(derivative@(effective_force*np.eye(points)-derivative*temp[None, :]))/effective_drag for temp in temperatures]
    coupling = switch_rate*np.eye(points)
    system = np.block([[operators[0]-coupling, coupling], [coupling, operators[1]-coupling]])
    system[-1, :] = 2*np.pi/points
    rhs = np.zeros(2*points)
    rhs[-1] = 1
    density = np.linalg.solve(system, rhs).reshape(2, points)
    current_x = np.array([(effective_force*rho-derivative@(temp*rho))/effective_drag for rho, temp in zip(density, temperatures)])
    current_y = (experiment['force_y']*density-magnetic*current_x)/gamma_y
    return temperatures, gradients, density, current_x, current_y, 2*np.pi/points


def heat_coefficient(friction, drag_ratio, magnetic):
    gamma_y = friction*drag_ratio
    relaxation = np.diag([3*friction, 2*friction+gamma_y, friction+2*gamma_y, 3*gamma_y])
    for i in range(3):
        relaxation[i, i+1] = -magnetic*np.sqrt((3-i)*(i+1))
        relaxation[i+1, i] = -relaxation[i, i+1]
    source = np.array([np.sqrt(6), 0., np.sqrt(2), 0.])
    return float(source@np.linalg.solve(relaxation, source)/4)


INPUT_FIELDS = ('force_x', 'force_y', 'drag_ratio', 'magnetic', 'temperature', 'contrast', 'wavenumber', 'bath_contrast', 'switch_rate')


@lru_cache(None)
def entropy_rate(force_x, force_y, drag_ratio, magnetic, temperature, contrast, wavenumber, bath_contrast, switch_rate, friction):
    e = dict(zip(INPUT_FIELDS, (force_x, force_y, drag_ratio, magnetic, temperature, contrast, wavenumber, bath_contrast, switch_rate)))
    temp, gradient, density, current_x, current_y, dx = positional_state(e, friction)
    gamma_y = friction*drag_ratio
    positional = np.sum((friction*current_x**2+gamma_y*current_y**2)/(temp*density))*dx
    positional += switch_rate*np.sum(density*np.log(density/density[::-1]))*dx
    thermal = heat_coefficient(friction, drag_ratio, magnetic)*np.sum(density*gradient**2/temp)*dx
RETURN_STATEMENT


def coefficients(experiments, friction=1.):
    return np.array([entropy_rate(*(e[field] for field in INPUT_FIELDS), friction) for e in experiments])


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
FIT_STATEMENT

    def predict(self, experiments):
        return coefficients(experiments, self.friction)
'''
fit = '''        inputs = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        coefficient = coefficients(inputs)
        inverse_friction = np.sum(coefficient*values/sigma**2)/np.sum(coefficient**2/sigma**2)
        self.friction = float(1/inverse_friction)
        return self'''
shortcut = common.replace('RETURN_STATEMENT', '    return float(positional+thermal)').replace('FIT_STATEMENT', fit)
oracle = common.replace('RETURN_STATEMENT', '''    ratio = temp/temp[::-1]
    contact = switch_rate*np.sum(density*(ratio-1-np.log(ratio)))*dx
    return float(positional+thermal+contact)''').replace('FIT_STATEMENT', fit)
starter = common.replace('RETURN_STATEMENT', '    return float(positional+thermal)').replace('FIT_STATEMENT', '        raise NotImplementedError("Fit the common friction coefficient from the calibration records.")')
(TASK/'solution/model.py').write_text(oracle)
(TASK/'environment/model.py').write_text(starter)
(STAGE/'scripts/entropy_anomaly_baseline.py').write_text(shortcut)

original = json.loads((REPO/'tasks/entropy-anomaly/environment/data/calibration.json').read_text())
for i, record in enumerate(original):
    record['input']['bath_contrast'] = 0.
    record['input']['switch_rate'] = float(.4+.1*(i%96%7))
for folder in ['environment', 'tests']:
    (TASK/folder/'data/calibration.json').write_text(json.dumps(original, indent=2)+'\n')
(TASK/'tests/metadata.json').write_text(json.dumps(dict(revision=5, prediction_limit=.04, parameter='friction', true_parameter=1.1), indent=2)+'\n')
