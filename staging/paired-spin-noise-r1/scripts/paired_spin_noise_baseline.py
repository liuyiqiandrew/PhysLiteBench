import numpy as np
from scipy.linalg import eigh
from scipy.special import expit


def energy_matrix(e):
    h=np.diag(np.array([-.35,.15,.7])+e['offset'])
    for i,j,w in [(0,1,1.),(1,2,.8),(0,2,.3)]:
        h[i,j]=h[j,i]=-e['hopping']*w
    h=np.kron(h,np.eye(2))
    pair=np.zeros((6,6),complex)
    for i,(d,b) in enumerate(zip([1.,.85,1.15],[0.,1.,-.4])):
        z=e['pairing']*d*np.exp(1j*e['phase']*b)
        pair[2*i,2*i+1]=z
        pair[2*i+1,2*i]=-z
    return np.block([[h,pair],[-pair.conj(),-h.T]])


def spectrum(e):
    energies,vectors=eigh(energy_matrix(e))
    occupation=expit(energies/e['temperature'])
    kernel=np.einsum('ar,br,r->abr',vectors,vectors.conj(),occupation)
    weights=np.zeros(6)
    weights[2*e['site']:2*e['site']+2]=[.5,-.5]
    coefficients=np.zeros((12,12),complex)
    for a in range(6):
        for b in range(6):
            coefficients+=weights[a]*weights[b]*np.outer(kernel[a+6,b+6],kernel[a,b])
    return energies[:,None]+energies[None,:],coefficients.real


def response(e):
    frequency,weight=spectrum(e)
    band=np.where(frequency>0.,frequency**2*np.exp(-.5*((frequency-e['center'])/e['width'])**2),0.)
    return float(np.sum(weight*band))


def predict_at(experiments, gain):
    return gain*np.array([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.gain=None

    def fit(self, records):
        records=list(records)
        x=np.array([response(r['input']) for r in records])
        y=np.array([r['value'] for r in records])
        w=1/np.array([r['sigma'] for r in records])**2
        self.gain=float(np.sum(w*x*y)/np.sum(w*x*x))
        return self

    def predict(self, experiments):
        return predict_at(experiments,self.gain)
