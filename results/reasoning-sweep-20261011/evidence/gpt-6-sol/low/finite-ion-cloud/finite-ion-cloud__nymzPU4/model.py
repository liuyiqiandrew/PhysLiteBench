import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre
from functools import lru_cache


@lru_cache(maxsize=None)
def coefficient(gamma,count,ratio):
    if count == 3:
        return three_ion_coefficient(gamma,ratio)
    length = np.log(ratio)
    if count == 1:
        slope = 2-gamma
        return -np.expm1(-2*length)*np.exp(slope*length)/(2*length*exprel(slope*length))
    a,b = 4-3*gamma,2-gamma
    def angular(gap):
        return hyp2f1(-gamma/2,-gamma/2,1,np.exp(-2*gap))
    partition = quad(lambda d: angular(d)*np.exp(b*d)*(length-d)*exprel(a*(length-d)),
                     0,length,epsabs=1e-11,epsrel=1e-11)[0]
    boundary = quad(lambda d: angular(d)*np.exp(b*d+a*(length-d)),
                    0,length,epsabs=1e-11,epsrel=1e-11)[0]
    return -np.expm1(-2*length)*boundary/(4*partition)


def predict_at(experiments,coupling):
    return np.asarray([coefficient(e['temperature_multiplier']*coupling,
                                  e['ion_count'],e['radius_ratio']) for e in experiments])


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        # The absolute disk size cancels after scaling all coordinates by a.
        groups = {}
        for record in records:
            e = record['input']
            key = (e['ion_count'], e['temperature_multiplier'], e['radius_ratio'])
            weight = 1 / record['sigma']**2
            total, weighted, sumsq = groups.get(key, (0., 0., 0.))
            groups[key] = (total + weight,
                           weighted + weight * record['value'],
                           sumsq + weight * record['value']**2)

        def objective(g):
            loss = 0.
            for (count, u, ratio), (weight, weighted, sumsq) in groups.items():
                prediction = coefficient(u*g, count, ratio)
                loss += weight*prediction**2 - 2*weighted*prediction + sumsq
            return loss

        result = minimize_scalar(objective, bounds=(0.7, 1.4),
                                 method='bounded', options={'xatol': 1e-11})
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma,ratio,order=64,angular_order=512):
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    # With angle differences alpha=theta_1-theta_2 and
    # beta=theta_2-theta_3, the third pair has angle alpha+beta.
    # The angular average is the sum of products of matching Fourier
    # coefficients of the three pair factors.
    angles = 2*np.pi*np.arange(angular_order)/angular_order
    cosines = np.cos(angles)
    def modes(gap):
        t = np.exp(-gap)
        factor = (1 - t[:,None])**2 + 2*t[:,None]*(1-cosines)
        return np.fft.rfft(factor**(gamma/2), axis=1).real/angular_order
    c1, c2, c3 = modes(d1), modes(d2), modes(d1+d2)
    angular = c1[:,0]*c2[:,0]*c3[:,0]
    angular += 2*np.sum(c1[:,1:-1]*c2[:,1:-1]*c3[:,1:-1], axis=1)
    angular += c1[:,-1]*c2[:,-1]*c3[:,-1]
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
