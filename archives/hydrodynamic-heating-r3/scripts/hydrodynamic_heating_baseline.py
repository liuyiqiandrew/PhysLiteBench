from functools import lru_cache
import numpy as np
import json
from scipy.optimize import minimize_scalar
GAMMA, NU = .06, .04

def modes(z, w, d, p):
    roots = np.roots([NU, GAMMA-1j*w-NU*w*w,
                     -w*w*(GAMMA-1j*w)-1j*w*p*p])
    vertical = np.sqrt(roots+0j)
    z = np.atleast_1d(z)
    out = np.zeros((len(z),4,4),complex)
    for j in range(2):
        current = (roots[j]-w*w)/(1j*w)
        for side in range(2):
            derivative = 1j*(1 if side == 0 else -1)*vertical[j]
            anchor = 0 if side == 0 else d
            wave = np.exp(derivative*(z-anchor))
            out[:,:,2*j+side] = wave[:,None]*np.array([1,derivative,current,derivative*current])
    return out

def modal_fields(e,p):
    w,d,alpha,tau = (e[k] for k in ('frequency','thickness','friction','relaxation'))
    left,right = modes([0,d],w,d,p)
    impedance=alpha/(1-1j*w*tau)
    boundary=np.array([left[1]+1j*w*left[0],right[1]-1j*w*right[0],
                       NU*left[3]-impedance*left[2],NU*right[3]+impedance*right[2]])
    coeff=np.linalg.solve(boundary,[2j*w,0,0,0])
    return lambda z: (modes(z,w,d,p)@coeff).T

@lru_cache(maxsize=2048)
def contact_response(frequency,thickness,friction,relaxation,plasma_frequency):
    e=dict(frequency=frequency,thickness=thickness,friction=friction,relaxation=relaxation)
    current=modal_fields(e,plasma_frequency)([0.])[2,0]
    traction=friction*current/(1-1j*frequency*relaxation)
    mean=float(np.real(traction*np.conjugate(current))/plasma_frequency**2)
    harmonic=traction*current/plasma_frequency**2
    return mean,harmonic


def predict_at(experiments,plasma_frequency):
    result=[]
    for experiment in experiments:
        e={k:experiment[k] for k in ('frequency','thickness','friction','relaxation')}
        mean,harmonic=contact_response(**e,plasma_frequency=plasma_frequency)
        result.append({'mean':mean,'in_phase':harmonic.real,'quadrature':harmonic.imag}[experiment['readout']])
    return np.asarray(result,dtype=float)


class Model:
    def __init__(self):
        self.plasma_frequency=None

    def fit(self,records):
        grouped={}
        for r in records:
            key=json.dumps(r['input'],sort_keys=True)
            if key not in grouped:
                grouped[key]=[r['input'],0.,0.]
            weight=1/float(r['sigma'])**2
            grouped[key][1]+=weight
            grouped[key][2]+=weight*float(r['value'])
        entries=list(grouped.values())
        inputs=[x[0] for x in entries]
        weights=np.array([x[1] for x in entries])
        values=np.array([x[2]/x[1] for x in entries])
        def objective(p):
            return float(np.sum(weights*(predict_at(inputs,p)-values)**2))
        fit=minimize_scalar(objective,method='bounded',bounds=(.85,1.15),options={'xatol':1e-12})
        if not fit.success:
            raise RuntimeError('Parameter fit failed.')
        self.plasma_frequency=float(min([.85,fit.x,1.15],key=objective))
        return self

    def predict(self,experiments):
        if self.plasma_frequency is None:
            raise ValueError('Fit plasma_frequency before prediction.')
        return predict_at(experiments,self.plasma_frequency)
