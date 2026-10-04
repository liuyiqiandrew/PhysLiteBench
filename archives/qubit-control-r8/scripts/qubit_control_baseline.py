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
    rates, factors, phases = [], [], []
    # A path is the spin basis state in the first and second waiting intervals.
    final, initial = np.indices((4,4))
    first, second = initial.ravel(), final.ravel()
    for e in experiments:
        psi = np.kron(*(rotation(r)[:,0] for r in e['preparation']))
        detector = np.kron(*(rotation(r).conj().T[:,0] for r in e['readout']))
        pulse = np.kron(*(rotation(r) for r in e['intermediate']))
        amplitude = psi[first]*pulse[second,first]*detector[second].conj()
        factors.append(np.outer(amplitude,amplitude.conj()))
        s = SIGNS@np.asarray(e['weights'])
        t1,t2 = e['waits']
        thermal = (np.ones(4) if e['temperature']==0 else
                   1/np.tanh(FREQUENCIES/(2*e['temperature'])))
        alpha1 = s[first,None]*np.sqrt(STRENGTHS)/FREQUENCIES*(np.exp(-1j*FREQUENCIES*t1)-1)
        alpha2 = s[second,None]*np.sqrt(STRENGTHS)/FREQUENCIES*(np.exp(-1j*FREQUENCIES*t2)-1)
        transported = alpha1*np.exp(-1j*FREQUENCIES*t2)
        alpha = alpha2+transported
        rates.append(.5*np.sum(thermal*abs(alpha[:,None,:]-alpha[None,:,:])**2,axis=-1))
        self_phase = np.sum(STRENGTHS/FREQUENCIES**2*(
            s[first,None]**2*(FREQUENCIES*t1-np.sin(FREQUENCIES*t1))+
            s[second,None]**2*(FREQUENCIES*t2-np.sin(FREQUENCIES*t2))),axis=-1)
        phase = self_phase[:,None]-self_phase[None,:]
        phases.append(phase)
    return np.array(rates),np.array(factors),np.array(phases)


def signal(terms, gamma):
    rates, factors, phases = terms
    return np.real(np.sum(factors*np.exp(-gamma*rates+1j*gamma*phases),axis=(1,2)))


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
