import numpy as np
from numpy.polynomial.legendre import leggauss
from functools import lru_cache
from scipy.optimize import minimize_scalar

@lru_cache(None)
def quadrature(order):
    nodes,weights=leggauss(order)
    return (nodes+1)/2,weights/2


def reflection_squared(q,xi,plasma,relaxation):
    epsilon=1+plasma**2/(xi*(xi+relaxation))
    inside=np.sqrt(q*q+plasma**2*xi/(xi+relaxation))
    te=((q-inside)/(q+inside))**2
    tm=((epsilon*q-inside)/(epsilon*q+inside))**2
    return te,tm


def pressure(experiment,plasma,order=72):
    a=experiment['separation'];temperature=experiment['temperature'];relaxation=experiment['relaxation']
    nodes,weights=quadrature(order)
    if temperature==0:
        y=40*nodes[:,None];u=nodes[None,:]**2
        q=y/(2*a);xi=q*u
        te,tm=reflection_squared(q,xi,plasma,relaxation)
        attenuation=np.exp(-y)
        integrand=y**3*(te*attenuation/(1-te*attenuation)+tm*attenuation/(1-tm*attenuation))
        integral=np.sum(integrand*(40*weights[:,None])*(2*nodes[None,:]*weights[None,:]))
        return -integral/(32*np.pi**2*a**4)
    spacing=4*np.pi*a*temperature
    positive=np.arange(1,max(2,int(np.ceil(40/spacing))+1))
    xi=2*np.pi*temperature*positive[:,None]
    y=spacing*positive[:,None]+40*nodes[None,:]
    q=y/(2*a)
    te,tm=reflection_squared(q,xi,plasma,relaxation)
    attenuation=np.exp(-y)
    integrand=y**2*(te*attenuation/(1-te*attenuation)+tm*attenuation/(1-tm*attenuation))
    total=float(np.sum(integrand*(40*weights[None,:])))
    y=40*nodes;q=y/(2*a);attenuation=np.exp(-y)
    transverse=((q-np.sqrt(q*q+plasma**2))/(q+np.sqrt(q*q+plasma**2)))**2
    zero=y*y*(attenuation/(1-attenuation)+transverse*attenuation/(1-transverse*attenuation))
    total+=.5*np.sum(zero*40*weights)
    return -temperature*total/(8*np.pi*a**3)


def predict_at(experiments,plasma,order=72):
    cache={}
    values=[]
    for e in experiments:
        key=(e['separation'],e['temperature'],e['relaxation'])
        if key not in cache:cache[key]=pressure(e,plasma,order)
        values.append(cache[key])
    return np.array(values)


class Model:
    def __init__(self):self.plasma_frequency=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def objective(plasma):
            residual=(predict_at(inputs,plasma)-values)/sigma
            return residual@residual
        self.plasma_frequency=float(minimize_scalar(objective,bounds=(4.,9.),method='bounded',options={'xatol':1e-10}).x)
        return self

    def predict(self,experiments):return predict_at(experiments,self.plasma_frequency)
