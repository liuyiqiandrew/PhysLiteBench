import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre
from functools import lru_cache


@lru_cache(maxsize=None)
def coefficient(gamma,count,ratio):
    """Dimensionless osmotic coefficient phi for a finite annulus.

    gamma is the coupling (u*g), count the number of mobile ions and ratio
    the finite outer radius R/a.  The central disk radius a drops out because
    every energy term is a logarithm of a ratio, so phi depends only on
    (gamma, N, R/a).
    """
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
                                  e['ion_count'],e['radius_ratio']) for e in experiments],
                      dtype=float)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        values = np.asarray([r['value'] for r in records],dtype=float)
        sigma = np.asarray([r['sigma'] for r in records],dtype=float)
        inputs = [r['input'] for r in records]
        def chi2(g):
            residual = (predict_at(inputs,g)-values)/sigma
            return float(np.dot(residual,residual))
        result = minimize_scalar(chi2,bounds=(0.7,1.4),method='bounded',
                                 options={'xatol':1e-10})
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


@lru_cache(maxsize=None)
def three_ion_coefficient(gamma,ratio,order=32,fourier=1024,nmodes=380):
    """Exact finite-cell osmotic coefficient for three mobile ions.

    The three-body angular integral does *not* factorise into a product of
    pairwise averages, so it is evaluated directly.  With the ordered radial
    variables (smallest logarithmic radius w, consecutive gaps d1,d2) the
    angular part is

        J(d1,d2) = int dphi dpsi (r12 r23 r13)**gamma,   radii 1,e^d1,e^{d1+d2},

    and the w-integral is analytic.  J is obtained from a Fourier convolution,
    which is exact because every factor depends on a single angular difference.
    """
    length = np.log(ratio)
    x,w = nodes(order)
    A,B = np.meshgrid(x,x,indexing='ij')
    WA,WB = np.meshgrid(w,w,indexing='ij')
    d1 = (length*A).ravel()
    d2 = (length*(1-A)*B).ravel()
    weight = (WA*WB*(length**2*(1-A))).ravel()
    span = length-d1-d2

    r23 = np.exp(d1)[:,None]
    r13 = np.exp(d1+d2)[:,None]
    theta = np.arange(fourier)*(2*np.pi/fourier)
    cos = np.cos(theta)[None,:]
    g12 = np.sqrt(1+r23**2-2*r23*cos)**gamma
    g23 = (r23**gamma)*np.sqrt(1+(r13/r23)**2-2*(r13/r23)*cos)**gamma
    g13 = np.sqrt(1+r13**2-2*r13*cos)**gamma
    c12 = np.fft.fft(g12,axis=1)/fourier
    c23 = np.fft.fft(g23,axis=1)/fourier
    c13 = np.fft.fft(g13,axis=1)/fourier
    ns = np.arange(fourier)
    n = np.where(ns<fourier//2,ns,ns-fourier)
    mask = np.abs(n)<=nmodes
    joint = (2*np.pi)**2*np.real(
        np.sum(c12[:,mask]*c23[:,mask]*c13[:,(fourier-ns)%fourier][:,mask],axis=1))

    slope = 6-6*gamma
    radial = np.exp((2-3*gamma)*(2*d1+d2))
    common = weight*radial*joint
    partition = np.dot(common,span*exprel(slope*span))
    boundary = np.dot(common,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
