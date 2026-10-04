import numpy as np

FREQUENCIES = np.array([1., 1.35, 1.9, 2.6])
STRENGTHS = np.array([.8, .6, .4, .3])
SIGNS = np.array([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])


def rotation(angles):
    phase, angle = angles
    c, s = np.cos(angle/2), np.sin(angle/2)
    return np.array([[c, -1j*s*np.exp(-1j*phase)],
                     [-1j*s*np.exp(1j*phase), c]])


def signal_terms(experiments):
    rates, factors = [], []

    for e in experiments:
        psi = np.kron(*(rotation(r)[:,0] for r in e['preparation']))
        detector = np.kron(*(rotation(r).conj().T[:,0] for r in e['readout']))
        factor = np.outer(psi,psi.conj())*np.outer(detector.conj(),detector)
        s = SIGNS@np.asarray(e['weights'])
        thermal = (np.ones(4) if e['temperature']==0 else
                   1/np.tanh(FREQUENCIES/(2*e['temperature'])))
        noise = np.sum(STRENGTHS*thermal*(1-np.cos(FREQUENCIES*e['wait']))/FREQUENCIES**2)
        rates.append((s[:,None]-s[None,:])**2*noise)
        factors.append(factor)

    return np.array(rates), np.array(factors)


def signal(terms, gamma):
    rates, factors = terms
    return np.real(np.sum(factors*np.exp(-gamma*rates),axis=(1,2)))


class QubitModel:
    def __init__(self):
        self.gamma = None

    def fit(self, runs):
        raise NotImplementedError

    def predict(self, experiments):
        if not experiments:
            return np.array([])
        return signal(signal_terms(experiments), self.gamma)
