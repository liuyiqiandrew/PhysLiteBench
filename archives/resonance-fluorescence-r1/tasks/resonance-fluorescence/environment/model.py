"""Stationary fluorescence gate statistics."""
import numpy as np
from scipy.linalg import expm

LOWER = np.array([[0.,1.],[0.,0.]])
NUMBER = LOWER.T@LOWER
TRACE = np.array([1.,0.,0.,1.])


def generator(rabi,detuning,decay):
    h=np.array([[0.,rabi/2],[rabi/2,detuning]])
    jump=decay*np.kron(LOWER,LOWER)
    liou=-1j*(np.kron(np.eye(2),h)-np.kron(h.T,np.eye(2)))
    liou+=jump-decay/2*(np.kron(np.eye(2),NUMBER)+np.kron(NUMBER.T,np.eye(2)))
    a=liou.copy();a[0]=TRACE
    rho=np.linalg.solve(a,np.array([1.,0.,0.,0.])).reshape((2,2),order='F')
    return liou,rho


def integrated_covariance(liou,seed,observable,gate):
    block=np.zeros((6,6),complex);block[:4,:4]=liou
    block[4,:4]=observable;block[5,4]=1
    initial=np.r_[seed.reshape(4,order='F'),0.,0.]
    return float((expm(gate*block)@initial)[5].real)


def response(rabi,detuning,decay,gate,efficiency):
    liou,rho=generator(rabi,detuning,decay)
    p=float(np.trace(NUMBER@rho).real);mean=efficiency*decay*p*gate
    seed=(NUMBER@rho+rho@NUMBER)/2-p*rho
    integral=integrated_covariance(liou,seed,NUMBER.T.reshape(4,order='F'),gate)
    variance=mean+2*(efficiency*decay)**2*integral
    return mean,variance



def predict_at(experiments, efficiency):
    values = []
    for e in experiments:
        if e['observable']=='mean':
            population = e['rabi']**2/(e['decay']**2+2*e['rabi']**2+4*e['detuning']**2)
            values.append(efficiency*e['decay']*population*e['duration'])
        else:
            values.append(response(e['rabi'],e['detuning'],e['decay'],e['duration'],efficiency)[1])
    return np.array(values)


class Model:
    def __init__(self):
        self.efficiency = None

    def fit(self, records):
        raise NotImplementedError('Fit efficiency from the calibration records.')

    def predict(self, experiments):
        if self.efficiency is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments, self.efficiency)
