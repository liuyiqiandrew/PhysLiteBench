from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq


@lru_cache(None)
def quadrature(order):
    return leggauss(order)


def potential(q, delta):
    return q**4/4 - q*q/2 + delta*q


@lru_cache(maxsize=32768)
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
def mean_energy(action_scale, delta_final, path_slope, s_final, order=24):
    def data(s):
        delta = delta_final+path_slope*np.log(s/s_final)
        return delta, geometry(float(delta))[2]

    z, w = quadrature(order)
    result = 0.
    for action, weight in zip(action_scale*(1.1+.1*z), w/2):
        crossing = brentq(lambda s: s**1.5*data(s)[1].sum()-2*np.pi*action,
                          .05, s_final, xtol=1e-13)
        delta, areas = data(crossing)
        branch_areas = crossing**1.5*areas/s_final**1.5
        probabilities = areas/areas.sum()
        energies = np.array([s_final**2*branch_energy(branch_areas[i], delta_final, i)
                             for i in (0, 1)])
        result += weight*np.dot(probabilities, energies)
    return float(result)


def predict_at(experiments, action_scale):
    return np.asarray([mean_energy(float(action_scale), float(e['delta_final']),
                                  float(e['path_slope']), float(e['s_final']))
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
            self.action_scale = float(minimize_scalar(objective, bounds=(.12, .16),
                                                     method='bounded', options={'xatol': 1e-12}).x)
        else:
            weights = 1/np.asarray([r['sigma'] for r in records])**2
            observed = float(weights@np.asarray([r['value'] for r in records])/weights.sum())
            response = lambda a: float(predict_at([first], a)[0])
            lower, upper = response(.12), response(.16)
            if observed <= lower:
                self.action_scale = .12
            elif observed >= upper:
                self.action_scale = .16
            else:
                self.action_scale = float(brentq(lambda a: response(a)-observed, .12, .16, xtol=2e-12))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.action_scale)
