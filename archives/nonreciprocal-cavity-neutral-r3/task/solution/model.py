import numpy as np
from scipy.optimize import minimize_scalar

PORT_RATES=np.array([1.,1.5,.8])


def response_matrices(experiments,loss_rate):
    count=len(experiments)
    h=np.broadcast_to(np.diag([-.3,.2,.6]),(count,3,3)).astype(complex).copy()
    phase=np.array([e['flux_phase'] for e in experiments])
    h[:,0,1]=h[:,1,0]=h[:,1,2]=h[:,2,1]=.8
    h[:,2,0]=.8*np.exp(1j*phase);h[:,0,2]=h[:,2,0].conj()
    frequency=np.array([e['frequency'] for e in experiments])
    generator=1j*(h-frequency[:,None,None]*np.eye(3))+.5*np.diag(PORT_RATES+loss_rate)
    resolvent=np.linalg.inv(generator)
    coupling=np.sqrt(PORT_RATES)
    scattering=np.eye(3)-coupling[None,:,None]*resolvent*coupling[None,None,:]
    frequencies,modes=np.linalg.eigh(h)
    transfer=np.sqrt(loss_rate)*coupling[None,:,None]*(resolvent@modes)
    return scattering,modes,transfer


def scattering_matrices(experiments,loss_rate):
    return response_matrices(experiments,loss_rate)[0]


def predict_at(experiments,loss_rate):
    if not experiments:return np.array([])
    scattering,modes,transfer=response_matrices(experiments,loss_rate)
    external=np.array([e['external_occupation'] for e in experiments])
    background=np.einsum('nij,nj->ni',abs(scattering)**2,external)
    occupation=np.array([e['body_occupation'] for e in experiments])
    noise=modes.conj().transpose(0,2,1)@(occupation[:,:,None]*modes)
    emission=np.einsum('nij,njk,nik->ni',transfer,noise,transfer.conj()).real
    output=background+emission
    port=np.array([e['output_port'] for e in experiments])
    selected=output[np.arange(len(experiments)),np.maximum(port,0)]
    return np.where(port==-1,np.sum(output,axis=1),selected)


class Model:
    def __init__(self):self.loss_rate=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        objective=lambda rate:np.sum(((predict_at(inputs,rate)-values)/sigma)**2)
        self.loss_rate=float(minimize_scalar(objective,bounds=(.1,.6),method='bounded',options={'xatol':1e-12}).x)
        return self

    def predict(self,experiments):return predict_at(experiments,self.loss_rate)
