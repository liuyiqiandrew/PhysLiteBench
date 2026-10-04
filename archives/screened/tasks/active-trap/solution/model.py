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


def characteristic(experiment, rotational_diffusion, modes=20, cutoff=25., tolerance=2e-10):
    from scipy.integrate import solve_ivp
    speed = float(experiment['speed'])
    diffusion = float(experiment['diffusion'])
    trap = float(experiment['trap_rate'])
    q = float(experiment['wavenumber'])
    if q*speed == 0:
        return float(np.exp(-diffusion*q*q/(2*trap)))
    harmonics = np.arange(-modes, modes+1)
    damping = np.diag(-rotational_diffusion*harmonics**2).astype(complex)
    coupling = np.diag(np.ones(2*modes), 1)+np.diag(np.ones(2*modes), -1)
    def generator(time):
        return damping + .5j*q*speed*np.exp(-trap*time)*coupling
    initial = np.zeros(2*modes+1, complex)
    initial[modes] = 1
    solution = solve_ivp(lambda t, y: generator(t)@y, (0, cutoff/trap), initial,
                         method='BDF', jac=lambda t, y: generator(t),
                         rtol=tolerance, atol=tolerance*.01)
    if not solution.success:
        raise RuntimeError(solution.message)
    return float(solution.y[modes, -1].real*np.exp(-diffusion*q*q/(2*trap)))


def predict_one(experiment, rotational_diffusion):
    if experiment['readout'] == 'trap_fourier':
        return characteristic(experiment, rotational_diffusion)
    return second_moment(experiment, rotational_diffusion)
