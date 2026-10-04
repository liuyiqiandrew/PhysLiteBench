import numpy as np
from functools import lru_cache


def hamiltonian(x, y, mass, handedness):
    x,y=np.broadcast_arrays(x,y)
    h=np.zeros(x.shape+(2,2),complex)
    z=mass+np.cos(x)+np.cos(y)
    h[...,0,0]=4.5+z;h[...,1,1]=4.5-z
    h[...,0,1]=np.sin(x)-1j*handedness*np.sin(y)
    h[...,1,0]=h[...,0,1].conj()
    return h


def frequency(experiment):
    x,y=experiment['wavevector']
    return float(np.linalg.eigvalsh(hamiltonian(x,y,experiment['mass'],experiment['handedness']))[experiment['band']])


@lru_cache(None)
def band_data(mass, handedness, order=96):
    k=-np.pi+2*np.pi*(np.arange(order)+.5)/order
    x,y=np.meshgrid(k,k,indexing='ij')
    h=hamiltonian(x,y,mass,handedness)
    energies,vectors=np.linalg.eigh(h)
    dx=np.zeros_like(h);dy=np.zeros_like(h)
    dx[...,0,0]=-np.sin(x);dx[...,1,1]=np.sin(x)
    dx[...,0,1]=dx[...,1,0]=np.cos(x)
    dy[...,0,0]=-np.sin(y);dy[...,1,1]=np.sin(y)
    dy[...,0,1]=-1j*handedness*np.cos(y);dy[...,1,0]=dy[...,0,1].conj()
    dagger=vectors.conj().swapaxes(-2,-1)
    vx=dagger@dx@vectors;vy=dagger@dy@vectors
    gaps=(energies[..., :,None]-energies[...,None,:])**2
    gaps[...,0,0]=gaps[...,1,1]=np.inf
    response=2*np.imag(vx*vy.swapaxes(-2,-1))/gaps
    mean_energy=(energies[..., :,None]+energies[...,None,:])/2
    energy_response=np.sum(response*mean_energy**2,axis=-1)
    return energies,np.sum(response,axis=-1),energy_response


def thermal_response(experiment, scale, order=96):
    energy,particle_response,energy_response=band_data(experiment['mass'],experiment['handedness'],order)
    temperature=experiment['temperature']
    population=1/np.expm1(scale*energy/temperature)
    return float(scale**2/temperature*np.mean(np.sum(population*energy_response,axis=-1)))


def predict_at(experiments, scale):
    return np.array([scale*frequency(e) if e['observable']=='frequency' else thermal_response(e,scale) for e in experiments])


class Model:
    def __init__(self):
        self.exchange_scale=None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        return predict_at(experiments,self.exchange_scale)
