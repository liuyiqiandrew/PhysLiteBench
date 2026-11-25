"""Independent causal Duhamel-memory quadrature at positive switch rates."""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn

TRUE_PARAMETER = 1.07


def characteristic(s):
    if abs(s) < 1e-3:
        return 1-s*s/14+s**4/504-s**6/33264
    return 15*spherical_jn(2,s)/(s*s)


@lru_cache(maxsize=None)
def memory(zeta, switch_rate):
    def kernel(s):
        return s*characteristic(s)*np.exp(-switch_rate*s)
    if zeta:
        real=quad(kernel,0,np.inf,weight='cos',wvar=zeta,
                  epsabs=1e-10,limlst=500,limit=500)[0]
        imag=quad(kernel,0,np.inf,weight='sin',wvar=zeta,
                  epsabs=1e-10,limlst=500,limit=500)[0]
    else:
        real=quad(kernel,0,np.inf,epsabs=1e-10,limit=1800)[0]
        imag=0.
    return real+1j*imag


def response(experiment,density,refinement=1.):
    kw=experiment['wavenumber']*experiment['velocity_width']
    z=experiment['frequency']/kw
    rates=refinement*np.asarray([.012,.006,.003])
    values=np.asarray([1/(1+density*memory(z,float(rate))/kw**2) for rate in rates])
    return complex(np.polynomial.polynomial.polyfit(rates,values,2)[0])


def predict(experiments,density,refinement=1.):
    return np.asarray([response(e,density,refinement).real for e in experiments])


def experiment(k,width,z):
    return {'wavenumber':float(k),'velocity_width':float(width),'frequency':float(k*width*z)}


def calibration_inputs():
    unique=[experiment(k,w,0.) for k in [1.5,1.85,2.2] for w in [.9,1.2]]
    unique += [experiment(k,w,z) for k,w in [(1.5,1.),(1.75,1.2),(2.,.9),(2.2,1.1)] for z in [1.5,1.9,2.3]]
    return unique*16


def hidden_inputs():
    geometry=[(1.5,.9),(1.6,1.),(1.75,.95),(1.9,1.)]
    return {
      'slow_phase': [experiment(k,w,z) for (k,w),z in zip(geometry,[.25,.3,.35,.4])],
      'middle_phase': [experiment(k,w,z) for (k,w),z in zip(geometry,[.42,.47,.52,.57])],
      'fast_phase': [experiment(k,w,z) for (k,w),z in zip(geometry,[.60,.615,.635,.65])],
      'nonresonant_anchors': [experiment(1.7,1.05,0.),experiment(1.9,.95,1.6),experiment(2.1,1.1,2.1),experiment(1.6,1.15,2.5)]}
