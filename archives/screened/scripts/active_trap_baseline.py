import numpy as np
from scipy.optimize import minimize_scalar


def second_moment(experiment, rotational_diffusion):
    speed = float(experiment['speed'])
    diffusion = float(experiment['diffusion'])
    rate = float(rotational_diffusion)
    if experiment['readout'] == 'free_msd':
        time = float(experiment['time'])
        return 4*diffusion*time + 2*speed**2*(rate*time + np.expm1(-rate*time))/rate**2
    trap = float(experiment['trap_rate'])
    return diffusion/trap + speed**2/(2*trap*(trap+rate))


class Model:
    def __init__(self):
        self.rotational_diffusion = .7

    def fit(self, records):
        experiments = [record['input'] for record in records]
        values = np.array([record['value'] for record in records])
        sigma = np.array([record['sigma'] for record in records])
        def objective(rate):
            prediction = np.array([second_moment(e, rate) for e in experiments])
            return float(np.sum(((prediction-values)/sigma)**2))
        result = minimize_scalar(objective, bounds=(.4, 1.1), method='bounded',
                                 options={'xatol': 1e-12})
        self.rotational_diffusion = float(result.x)
        return self

    def predict(self, experiments):
        return np.array([predict_one(e, self.rotational_diffusion) for e in experiments])


def predict_one(experiment, rotational_diffusion):
    variance = second_moment(experiment, rotational_diffusion)
    if experiment['readout'] != 'trap_fourier':
        return variance
    q = float(experiment['wavenumber'])
    return float(np.exp(-.5*q*q*variance))
