import numpy as np
from scipy.optimize import minimize_scalar

PAULI = [np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1.])]
FIRST = [np.kron(s,np.eye(2))/2 for s in PAULI]
SECOND = [np.kron(np.eye(2),s)/2 for s in PAULI]
EXCHANGE = sum(a@b for a,b in zip(FIRST,SECOND))
MAGNETIZATION = FIRST[2]+SECOND[2]
PARALLEL = np.diag([1.,0.,0.,1.])
SINGLET = np.array([0.,1.,-1.,0.])/np.sqrt(2.)
TRIPLET = np.eye(4)-np.outer(SINGLET,SINGLET)
OBSERVABLES = dict(magnetization=MAGNETIZATION,parallel_probability=PARALLEL,spin_correlation=EXCHANGE)


class Model:
    def __init__(self):
        self.temperature = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(temperature):
            self.temperature = temperature
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        result = minimize_scalar(loss,bounds=(.3,1.5),method='bounded',options={'xatol':1e-12})
        self.temperature = float(result.x)
        return self

    def predict(self, experiments):
        fields = np.array([e['field'] for e in experiments])
        hamiltonian = .9*EXCHANGE-fields[:,None,None]*MAGNETIZATION
        energies,vectors = np.linalg.eigh(hamiltonian)
        weights = np.exp(-(energies-energies.min(axis=1)[:,None])/self.temperature)
        density = (vectors*weights[:,None,:])@vectors.conj().transpose(0,2,1)
        out = []
        for e,rho in zip(experiments,density):
            detector = TRIPLET if e['selection']=='triplet' else np.eye(4)
            selected = detector@rho@detector
            out.append(float(np.trace(selected@OBSERVABLES[e['observable']]).real/np.trace(selected).real))
        return np.array(out)
