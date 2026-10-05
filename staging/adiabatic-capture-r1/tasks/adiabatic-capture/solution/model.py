from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq


@lru_cache(None)
def quadrature(order):
    return leggauss(order)


def potential(q, delta):
    return q**4/4 - q*q/2 + delta*q


def phase_volume(energy, delta, order=96):
    roots = np.roots([.25, 0., -.5, delta, -energy])
    roots = np.sort(roots.real[abs(roots.imag) < 1e-8])
    if len(roots) < 2:
        return 0.
    z, w = quadrature(order)
    theta = np.pi*z/2
    total = 0.
    for left, right in zip(roots[::2], roots[1::2]):
        q = (right+left)/2 + (right-left)*np.sin(theta)/2
        jacobian = (right-left)*np.cos(theta)*np.pi/4
        momentum = np.sqrt(np.maximum(0., 2*(energy-potential(q, delta))))
        total += 2*np.sum(w*jacobian*momentum)
    return float(total)


def energy_for_volume(volume, delta):
    extrema = np.sort(np.roots([1., 0., -1., delta]).real)
    lower = float(min(potential(extrema[[0, 2]], delta))) + 1e-12
    upper = 1.
    while phase_volume(upper, delta) < volume:
        upper = 2*upper+1
    return float(brentq(lambda energy: phase_volume(energy, delta)-volume,
                        lower, upper, xtol=2e-12))


@lru_cache(None)
def geometry(delta):
    extrema = np.sort(np.roots([1.,0.,-1.,delta]).real)
    saddle = extrema[1]
    barrier = potential(saddle,delta)
    span = np.sqrt(2*(1-saddle*saddle))
    left,right = -saddle-span,-saddle+span
    z,w = quadrature(128)
    theta = np.pi*z/2
    areas=[]
    for lo,hi in [(left,saddle),(saddle,right)]:
        q=(lo+hi)/2+(hi-lo)*np.sin(theta)/2
        jacobian=(hi-lo)*np.cos(theta)*np.pi/4
        momentum=abs(q-saddle)*np.sqrt(np.maximum(0.,(right-q)*(q-left)))/np.sqrt(2)
        areas.append(float(2*np.sum(w*jacobian*momentum)))
    return extrema,float(barrier),np.asarray(areas)


def branch_volume(energy,delta,branch,order=96):
    roots=np.roots([.25,0.,-.5,delta,-energy])
    real=np.sort(roots.real[abs(roots.imag)<1e-8])
    saddle=geometry(delta)[0][1]
    z,w=quadrature(order); theta=np.pi*z/2; volume=0.
    for lo,hi in zip(real[::2],real[1::2]):
        if ((hi<saddle)!=(branch==0)):
            continue
        q=(hi+lo)/2+(hi-lo)*np.sin(theta)/2
        jacobian=(hi-lo)*np.cos(theta)*np.pi/4
        momentum=np.sqrt(np.maximum(0.,2*(energy-potential(q,delta))))
        volume+=2*np.sum(w*jacobian*momentum)
    return float(volume)


def branch_energy(volume,delta,branch):
    extrema,barrier,areas=geometry(delta)
    lower=float(potential(extrema[0 if branch==0 else 2],delta))
    # Exact endpoint areas keep roots arbitrarily close to the separatrix bracketed.
    def residual(energy):
        if energy<=lower:
            return -volume
        if energy>=barrier:
            return float(areas[branch]-volume)
        return branch_volume(energy,delta,branch)-volume
    return float(brentq(residual,lower,barrier,xtol=2e-12))


@lru_cache(maxsize=16384)
def mean_energy(action_scale,delta,s_final,order=32):
    _,_,areas=geometry(delta)
    fractions=areas/areas.sum()
    lower,upper=action_scale,1.2*action_scale
    split=float(np.clip(areas.sum()*s_final**1.5/(2*np.pi),lower,upper))
    z,w=quadrature(order); result=0.
    for lo,hi,captured in [(lower,split,True),(split,upper,False)]:
        if hi<=lo:
            continue
        actions=(lo+hi)/2+(hi-lo)*z/2
        for action,weight in zip(actions,w*(hi-lo)/(2*(upper-lower))):
            area=2*np.pi*action/s_final**1.5
            if captured:
                e=sum(fractions[i]*branch_energy(fractions[i]*area,delta,i) for i in (0,1))
            else:
                e=energy_for_volume(area,delta)
            result+=weight*s_final*s_final*e
    return float(result)


def predict_at(experiments, action_scale):
    return np.asarray([mean_energy(float(action_scale), float(e['delta']), float(e['s_final']))
                       for e in experiments])


class Model:
    def __init__(self):
        self.action_scale = None

    def fit(self, records):
        inputs = [r['input'] for r in records]
        first = inputs[0]
        if any(e != first for e in inputs):
            from scipy.optimize import minimize_scalar
            y = np.asarray([r['value'] for r in records])
            sigma = np.asarray([r['sigma'] for r in records])
            objective = lambda a: float(np.sum(((predict_at(inputs, a)-y)/sigma)**2))
            self.action_scale = float(minimize_scalar(objective, bounds=(.12,.16),
                                                     method='bounded', options={'xatol':1e-12}).x)
        else:
            weight = 1/np.asarray([r['sigma'] for r in records])**2
            observed = float(weight@np.asarray([r['value'] for r in records])/weight.sum())
            response = lambda a: float(predict_at([first], a)[0])
            target = float(np.clip(observed, response(.12), response(.16)))
            self.action_scale = float(brentq(lambda a: response(a)-target, .12, .16, xtol=2e-12))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.action_scale)
