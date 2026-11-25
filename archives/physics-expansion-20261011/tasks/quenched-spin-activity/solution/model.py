import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter


def potential(angles, e):
    m=np.sin(angles)
    total=(m[...,0]+m[...,1])/2
    f=e['coupling']*total[...,None]+e['field']+np.array([e['disorder'],-e['disorder']])
    s=np.array([e['bias_plus'],e['bias_minus']])
    return .25*np.sum(np.exp(-s)*np.cos(angles)/np.cosh(f)-1+m*np.tanh(f),axis=-1)


def response(e, kappa, grid=49):
    if e['coupling'] == 0.:
        f=e['field']+np.array([e['disorder'],-e['disorder']])
        s=np.array([e['bias_plus'],e['bias_minus']])
        return float(kappa*.25*np.sum(-1+np.sqrt(np.tanh(f)**2+np.exp(-2*s)/np.cosh(f)**2)))
    a=np.linspace(-np.pi/2,np.pi/2,grid)
    points=np.stack(np.meshgrid(a,a,indexing='ij'),axis=-1)
    values=potential(points,e)
    mask=values >= maximum_filter(values,size=3,mode='constant',cval=-np.inf)
    starts=list(points[mask])
    starts.extend([points[np.unravel_index(np.argmax(values),values.shape)],np.zeros(2)])
    candidates=[]
    for x in starts:
        opt=minimize(lambda angles:-potential(angles,e),x,bounds=[(-np.pi/2,np.pi/2)]*2,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10})
        candidates.append((-opt.fun,np.sin(opt.x)))
    value,m=max(candidates,key=lambda row:row[0])
    return float(kappa*value)


def predict_at(experiments, attempt_rate):
    return np.asarray([response(e, float(attempt_rate)) for e in experiments])


class Model:
    def __init__(self):
        self.attempt_rate = None

    def fit(self, records):
        basis = predict_at([r['input'] for r in records], 1.)
        sigma = np.asarray([r['sigma'] for r in records])
        value = np.asarray([r['value'] for r in records])
        self.attempt_rate = float(np.sum(basis*value/sigma**2)/np.sum((basis/sigma)**2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.attempt_rate)
