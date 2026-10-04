import numpy as np
from scipy.special import softmax

FREQUENCIES = np.array([1., 1.35, 1.9, 2.6])
STRENGTHS = np.array([.8, .6, .4, .3])
SIGNS = np.array([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])


def rotation(angles):
    phase, angle = angles
    c, s = np.cos(angle/2), np.sin(angle/2)
    return np.array([[c, -1j*s*np.exp(-1j*phase)],
                     [-1j*s*np.exp(1j*phase), c]])


def signal_terms(experiments):
    energies, factors, rates, phases = [], [], [], []
    for e in experiments:
        s = SIGNS @ np.asarray(e['weights'])
        energies.append(s**2 * np.sum(STRENGTHS/FREQUENCIES) / e['temperature'])
        preparation = np.kron(*(rotation(r) for r in e['preparation']))
        detector = np.kron(*(rotation(r).conj().T[:,0] for r in e['readout']))
        amplitude = preparation * detector.conj()[:,None]
        factors.append(np.einsum('ir,jr->rij', amplitude, amplitude.conj()))
        difference = s[:,None] - s[None,:]
        squared_difference = s[:,None]**2 - s[None,:]**2
        time = e['wait']
        thermal = 1/np.tanh(FREQUENCIES/(2*e['temperature']))
        noise = np.sum(STRENGTHS*thermal*(1-np.cos(FREQUENCIES*time))/FREQUENCIES**2)
        motion_phase = np.sum(STRENGTHS*(FREQUENCIES*time-np.sin(FREQUENCIES*time))/FREQUENCIES**2)
        displacement_phase = np.sum(STRENGTHS*np.sin(FREQUENCIES*time)/FREQUENCIES**2)
        rates.append(noise*difference**2)
        phases.append(motion_phase*squared_difference[None,:,:]
                      + 2*displacement_phase*s[:,None,None]*difference[None,:,:])
    return np.array(energies), np.array(factors), np.array(rates), np.array(phases)


def signal(terms, gamma):
    energies, factors, rates, phases = terms
    populations = softmax(gamma*energies, axis=1)
    conditional = np.exp(-gamma*rates[:,None,:,:] + 1j*gamma*phases)
    return np.real(np.einsum('nr,nrij,nrij->n', populations, factors, conditional))


class QubitModel:
    def __init__(self):
        self.gamma = None

    def fit(self, runs):
        from scipy.optimize import minimize_scalar
        terms = signal_terms([r['experiment'] for r in runs])
        values = np.array([r['probability'] for r in runs])
        sigma = np.array([r['sigma'] for r in runs])
        def objective(gamma):
            residual = (signal(terms,gamma)-values)/sigma
            return float(residual@residual)
        result = minimize_scalar(objective,bounds=(.04,.25),method='bounded',
                                 options={'xatol':1e-13})
        self.gamma = float(result.x)
        return self

    def predict(self, experiments):
        if not experiments:
            return np.array([])
        return signal(signal_terms(experiments), self.gamma)
