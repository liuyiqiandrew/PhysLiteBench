import numpy as np
from scipy.optimize import minimize_scalar

THERMAL_ENERGY = 1.380649e-5 * 300.  # pN micrometer
RADIUS = .5  # micrometer
STIFFNESS = .3  # pN / micrometer


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        inputs = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(log_eta):
            self.viscosity = float(np.exp(log_eta))
            return np.sum(((self.predict(inputs)-values)/sigma)**2)
        result = minimize_scalar(loss, bounds=np.log([.0005, .003]), method='bounded',
                                 options={'xatol': 1e-12})
        self.viscosity = float(np.exp(result.x))
        return self

    def predict(self, experiments):
        if not experiments:
            return np.empty(0)
        k = STIFFNESS*np.array([e['factors'] for e in experiments])
        root_k = np.sqrt(k)
        n = len(experiments)
        mobility = np.zeros((n, 2, 2))
        mobility[:, 0, 0] = mobility[:, 1, 1] = 1/(6*np.pi*self.viscosity*RADIUS)
        mobility[:, 0, 1] = mobility[:, 1, 0] = 1/(4*np.pi*self.viscosity*np.array([e['separation'] for e in experiments]))
        symmetric_rate = root_k[:, :, None]*mobility*root_k[:, None, :]
        rates, modes = np.linalg.eigh(symmetric_rate)
        time = np.array([e['time'] for e in experiments])
        evolution_y = (modes*np.exp(-rates*time[:, None])[:, None, :])@modes.transpose(0, 2, 1)
        evolution_x = evolution_y*root_k[:, None, :]/root_k[:, :, None]
        shifts = np.array([e['shift'] for e in experiments])
        means = shifts-np.einsum('nij,nj->ni', evolution_x, shifts)
        diffusion = 2*THERMAL_ENERGY*mobility*np.eye(2)[None, :, :]
        diffusion_y = root_k[:, :, None]*diffusion*root_k[:, None, :]
        projected = modes.transpose(0, 2, 1)@diffusion_y@modes
        sums = rates[:, :, None]+rates[:, None, :]
        covariance_modes = projected*(-np.expm1(-sums*time[:, None, None]))/sums
        covariance = (modes@covariance_modes@modes.transpose(0, 2, 1))/root_k[:, :, None]/root_k[:, None, :]
        return np.array([means[j, e['bead']] if e['measurement']=='mean' else covariance[j, e['pair'][0], e['pair'][1]]
                         for j, e in enumerate(experiments)])
