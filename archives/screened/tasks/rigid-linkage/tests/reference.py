"""Cartesian velocity Jacobian, Gaussian canonical momenta, Legendre angular quadrature."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss

PARAMETER = 'stiffness'
TRUE_PARAMETER = 1.2


def experiment(locked,factors=(1.,1.),bias=(0.,0.),mass_ratio=.1,harmonic=(2,-2),phase=0.):
    return dict(locked=locked,factors=list(factors),bias=list(bias),mass_ratio=float(mass_ratio),
                harmonic=list(harmonic),phase=float(phase))


def calibration_inputs():
    return [experiment(True,factors=(.7,float(f)),harmonic=(0,1)) for f in np.linspace(.2,2.,100)]


def hidden_inputs():
    return {
        'unforced_shape': [experiment(False,factors=(0.,0.),mass_ratio=r) for r in np.geomspace(.02,1.,24)],
        'aligned_fields': [experiment(False,factors=(f,f),mass_ratio=.05) for f in np.linspace(.2,1.6,24)],
        'turned_field': [experiment(False,factors=(1.5,.8),bias=(0.,b),mass_ratio=.15) for b in np.linspace(-np.pi,np.pi,24)]}


def inertia(first,second,ratio):
    """Pull back Cartesian point-mass kinetic energy to the two angles."""
    first,second = np.broadcast_arrays(first,second)
    jacobian = np.zeros(first.shape+(4,2))
    jacobian[...,0,0] = -np.sin(first)
    jacobian[...,1,0] = np.cos(first)
    jacobian[...,2,0] = -np.sin(first)
    jacobian[...,3,0] = np.cos(first)
    jacobian[...,2,1] = -np.sin(second)
    jacobian[...,3,1] = np.cos(second)
    masses = np.array([ratio,ratio,1.,1.])
    return np.einsum('...ki,k,...kj->...ij',jacobian,masses,jacobian)


@lru_cache(40)
def quadrature(locked,ratio,order):
    nodes,weights = leggauss(order)
    nodes = np.pi*nodes
    if locked:
        return 0.,nodes,weights
    first,second = np.meshgrid(nodes,nodes,indexing='ij')
    metric = inertia(first,second,ratio)
    inverse = np.linalg.inv(metric)
    # Transform the canonical-momentum Gaussian by its precision Cholesky factor.
    # Its integral, aside from angle-independent 2*pi, is 1/det(cholesky(G^-1)).
    precision_factor = np.linalg.cholesky(inverse)
    momentum_integral = 1/(precision_factor[...,0,0]*precision_factor[...,1,1])
    return first,second,weights[:,None]*weights[None,:]*momentum_integral


def predict(experiments,stiffness,order=192):
    out = []
    for e in experiments:
        first,second,measure = quadrature(e['locked'],e['mass_ratio'],order)
        potential = stiffness*(e['factors'][0]*(1-np.cos(first-e['bias'][0]))
                               +e['factors'][1]*(1-np.cos(second-e['bias'][1])))
        density = measure*np.exp(-potential)
        value = np.cos(e['harmonic'][0]*first+e['harmonic'][1]*second+e['phase'])
        out.append(np.sum(density*value)/np.sum(density))
    return np.array(out)
