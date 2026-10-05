"""Bounded scientific feasibility; no task packaging or model-agent calls."""
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from numpy.polynomial.legendre import leggauss


def langevin(z):
    if abs(z) < 1e-3:
        return z/3-z**3/45+2*z**5/945
    return 1/np.tanh(z)-1/z


def mean_extension(n, b, temperature, force):
    return n*b*langevin(force*b/temperature)


def extension_derivative(n, b, temperature, force):
    z = force*b/temperature
    return n*(1/np.tanh(z)-z/np.sinh(z)**2)


def source_force(n, b, temperature, extension):
    r = extension/(n*b)
    if r == 0:
        return 0.
    z = brentq(lambda z: langevin(z)-r, 1e-12, 2/(1-r), xtol=1e-13)
    return temperature*z/b


def projection_density(n, b, extension):
    s = (n-abs(extension)/b)/2
    if s <= 0:
        return 0.
    return math.fsum((-1)**k*math.comb(n, k)*(s-k)**(n-1)
                     for k in range(min(n, int(s))+1))/math.factorial(n-1)/(2*b)


def physical_force(n, b, temperature, extension):
    if extension == 0:
        return 0.
    s = (n-abs(extension)/b)/2
    terms = range(min(n, int(s))+1)
    f = math.fsum((-1)**k*math.comb(n, k)*(s-k)**(n-1) for k in terms)
    df = (n-1)*math.fsum((-1)**k*math.comb(n, k)*(s-k)**(n-2) for k in terms)
    return math.copysign(temperature*df/(2*b*f), extension)


@lru_cache(None)
def fourier_nodes(panels, order):
    z, w = leggauss(order)
    left = np.arange(panels)*np.pi
    q = (left[:, None]+np.pi/2*(1+z)[None, :]).ravel()
    weights = np.tile(np.pi*w/2, panels)
    return q, weights


def fourier_reference(n, b, temperature, extension, panels=256, order=32,
                       clamp_stiffness=None):
    """Product characteristic function, independent of alternating finite sums.

    A finite clamp multiplies the Fourier integrand by the transform of its
    Boltzmann Gaussian. Its force is the mechanical mean K*(X-<x_end>).
    """
    q, weights = fourier_nodes(panels, order)
    phi = np.sinc(q/np.pi)**n
    if clamp_stiffness is not None:
        phi *= np.exp(-temperature*q*q/(2*clamp_stiffness*b*b))
    phase = q*extension/b
    denominator = np.dot(weights, phi*np.cos(phase))
    numerator = np.dot(weights, q*phi*np.sin(phase))
    return float(temperature/b*numerator/denominator)


def force_calibration_reference(n, b, temperature, force, order=96):
    u, w = leggauss(order)
    weights = w*np.exp(force*b*u/temperature)
    return float(n*b*np.dot(weights, u)/weights.sum())


def main():
    start = time.perf_counter()
    corners = []
    for n,b,t,per_link in itertools.product([4,6,8,10,12], [.8,1.2], [.6,1.4], [.15,.6]):
        x = n*per_link
        f = physical_force(n,b,t,x)
        s = source_force(n,b,t,x)
        ref = fourier_reference(n,b,t,x)
        fine = fourier_reference(n,b,t,x,512,48)
        corners.append({'links':n,'length':b,'temperature':t,'extension':x,
                        'physical':f,'source':s,'relative_gap':abs(s-f)/abs(f),
                        'reference_error':abs(ref-f),'reference_refinement':abs(ref-fine)})
    rng = np.random.default_rng(410091)
    random_rows = []
    for _ in range(24):
        n = int(rng.integers(4,13)); b=rng.uniform(.8,1.2)
        t=rng.uniform(.6,1.4); x=n*rng.uniform(.15,.6)
        f=physical_force(n,b,t,x); ref=fourier_reference(n,b,t,x)
        random_rows.append({'links':n,'length':b,'temperature':t,'extension':x,
                            'physical':f,'reference_error':abs(f-ref),
                            'relative_gap':abs(source_force(n,b,t,x)-f)/f})
    soft_clamp = []
    for n,b,t,p in [(4,.8,1.4,.6),(4,1.2,.6,.15),(8,1.,1.,.4),(12,.8,.6,.6)]:
        x=n*p; exact=physical_force(n,b,t,x)
        values=[fourier_reference(n,b,t,x,256,40,k) for k in [100,200,400,800]]
        extrap=(values[0]-6*values[1]+8*values[2])/3
        refined=(values[1]-6*values[2]+8*values[3])/3
        soft_clamp.append({'links':n,'length':b,'temperature':t,'extension':x,
                           'physical':exact,'stiffnesses':[100,200,400,800],
                           'forces':values,'extrapolation_error':abs(extrap-exact),
                           'refined_extrapolation_error':abs(refined-exact)})
    cal = list(itertools.product([4,8,12],[.6,1.,1.4],[.3,1.,3.]))
    cal_reference_error=0.; derivative_min=float('inf'); derivative_fd_error=0.
    recovery=[]
    for b in np.linspace(.8,1.2,25):
        y=np.array([mean_extension(n,b,t,f) for n,t,f in cal])
        z=np.array([force_calibration_reference(n,b,t,f) for n,t,f in cal])
        cal_reference_error=max(cal_reference_error,float(np.max(abs(y-z))))
        for n,t,f in cal:
            derivative=extension_derivative(n,b,t,f)
            fd=(mean_extension(n,b+1e-5,t,f)-mean_extension(n,b-1e-5,t,f))/(2e-5)
            derivative_min=min(derivative_min,derivative)
            derivative_fd_error=max(derivative_fd_error,abs(fd-derivative))
        def loss(v):
            return np.mean((np.array([mean_extension(n,v,t,f) for n,t,f in cal])-y)**2)
        fit=minimize_scalar(loss,bounds=(.8,1.2),method='bounded',options={'xatol':1e-13})
        recovered=min([.8,float(fit.x),1.2],key=loss)
        recovery.append({'true_length':float(b),'fit':recovered,'absolute_error':abs(recovered-b)})
    # Density normalization and variance use compact real-space quadrature.
    density_checks=[]
    for n in [4,6,9,12]:
        knots=np.arange(-n,n+1,2,dtype=float)
        norm=quad(lambda x:projection_density(n,1.,x),-n,n,points=knots[1:-1],epsabs=1e-10)[0]
        variance=quad(lambda x:x*x*projection_density(n,1.,x),-n,n,points=knots[1:-1],epsabs=1e-10)[0]
        density_checks.append({'links':n,'normalization_error':abs(norm-1),
                               'variance_error':abs(variance-n/3)})
    # In the last support interval only the leading simplex term survives.
    endpoint_checks=[]
    for n in [4,6,8,12]:
        x=n-0.4
        endpoint_checks.append(abs(physical_force(n,1.,1.,x)-(n-1)/(n-x)))
    # Large-N behavior tested by the independent smooth Fourier calculation.
    large_n=[]
    for n in [4,8,16,32,64,128]:
        exact=fourier_reference(n,1.,1.,n*.2,64,64)
        source=source_force(n,1.,1.,n*.2)
        large_n.append({'links':n,'physical':exact,'source':source,'relative_gap':abs(source-exact)/exact})
    # N=1 has flat projected density and zero holding force in its interior.
    n1_density=[projection_density(1,1.,x) for x in [-.8,0.,.8]]
    report={'status':'bounded_feasibility_only_no_model_evaluations','corners':corners,
            'random_checks':random_rows,'soft_clamp':soft_clamp,
            'calibration':{'settings':len(cal),'common_exact_formula':True,
                           'independent_quadrature_error':cal_reference_error,
                           'minimum_positive_derivative':derivative_min,
                           'derivative_fd_error':derivative_fd_error,'recoveries':recovery},
            'limits':{'density_checks':density_checks,'endpoint_error':max(endpoint_checks),
                      'large_n':large_n,'one_link_interior_density':n1_density},
            'summary':{'minimum_physical_force':min(r['physical'] for r in corners),
                       'minimum_corner_relative_gap':min(r['relative_gap'] for r in corners),
                       'maximum_reference_error':max(r['reference_error'] for r in corners+random_rows),
                       'maximum_reference_refinement':max(r['reference_refinement'] for r in corners),
                       'maximum_fit_error':max(r['absolute_error'] for r in recovery),
                       'runtime_seconds':time.perf_counter()-start}}
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(r['physical']>0 and r['source']>0 for r in corners)
    assert report['summary']['minimum_corner_relative_gap']>.04
    assert report['summary']['maximum_reference_error']<1e-6
    assert derivative_min>0 and cal_reference_error<1e-10
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
