from functools import lru_cache
import json
import numpy as np
from scipy.linalg import expm,eigh
from scipy.optimize import minimize_scalar

POISSON=.48
VISCOSITY=.15


@lru_cache(None)
def grid(points):
    x=np.arange(points)*2*np.pi/points
    modes=np.fft.fftfreq(points,1/points)
    derivative=np.fft.ifft(1j*modes[:,None]*np.fft.fft(np.eye(points),axis=0),axis=0).real
    return x,modes,derivative


def homogeneous_compliance(q):
    q=np.asarray(q,dtype=float)
    value=np.full_like(q,(1+POISSON)*(1-2*POISSON)/(1-POISSON))
    nonzero=q>1e-8;t=q[nonzero]
    numerator=(3-4*POISSON)*np.sinh(2*t)-2*t
    denominator=(3-4*POISSON)*np.cosh(2*t)+2*t*t+5-12*POISSON+8*POISSON**2
    value[nonzero]=2*(1-POISSON**2)*numerator/(t*denominator)
    return value


@lru_cache(256)
def surface_compliance(q,contrast,pattern_mode,pattern_phase,points=49):
    x,modes,derivative=grid(points)
    relative_modulus=1+contrast*np.cos(pattern_mode*x+pattern_phase)
    # Eliminate the bulk displacement/stress system through thin numerical
    # slices. Updating its compliance avoids growing evanescent modes.
    mu=1/(2*(1+POISSON));lam=POISSON/((1+POISSON)*(1-2*POISSON));longitudinal=lam+2*mu
    ratio=lam/longitudinal;d=derivative*q;zero=np.zeros((points,points))
    a=np.block([[zero,-d],[-ratio*d,zero]])
    b=np.block([[np.diag(1/(mu*relative_modulus)),zero],[zero,np.diag(1/(longitudinal*relative_modulus))]])
    stiffness=np.block([[-d@np.diag((longitudinal-lam*lam/longitudinal)*relative_modulus)@d,zero],[zero,zero]])
    generator=np.block([[a,b],[stiffness,-a.T]])
    slices=64;transfer=expm(generator/slices);size=2*points
    response=np.zeros((size,size))
    for _ in range(slices):
        numerator=transfer[:size,:size]@response+transfer[:size,size:]
        denominator=transfer[size:,:size]@response+transfer[size:,size:]
        response=np.linalg.solve(denominator.T,numerator.T).T
    normal=response[points:,points:]
    return (normal+normal.T)/2


@lru_cache(512)
def relaxation(key,points=49):
    e=json.loads(key)
    x,modes,derivative=grid(points)
    depth=e['thickness']*.001
    k=e['fundamental_wavenumber']*1000
    compliance=depth/1e6*surface_compliance(k*depth,e['contrast'],e['pattern_mode'],e['pattern_phase'],points)
    mobility=(e['gap']*1e-6)**3/(12*VISCOSITY)
    flow=-mobility*k*k*(derivative@derivative)
    rates,vectors=eigh(flow,compliance)
    # The one zero mode is uniform pressure; mean gap volume is conserved.
    rates=np.maximum(rates,0)
    initial=np.cos(e['pressure_mode']*x+e['pressure_phase'])
    if e['detector_mode']==0:weight=np.ones(points)/points
    else:weight=2*np.cos(e['detector_mode']*x+e['detector_phase'])/points
    amplitudes=(weight@compliance@vectors)*(vectors.T@compliance@initial)*1e9
    return rates,amplitudes


def predict_at(experiments,young_modulus):
    out=[]
    for e in experiments:
        setup={k:v for k,v in e.items() if k not in ['pressure','elapsed_time']}
        rates,amplitudes=relaxation(json.dumps(setup,sort_keys=True))
        out.append(e['pressure']/young_modulus*np.sum(amplitudes*np.exp(-young_modulus*rates*e['elapsed_time'])))
    return np.array(out)


class Model:
    def __init__(self):self.young_modulus=None

    def fit(self,records):
        experiments=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def objective(modulus):return np.sum(((predict_at(experiments,modulus)-values)/sigma)**2)
        self.young_modulus=float(minimize_scalar(objective,bounds=(.7,1.6),method='bounded',options={'xatol':1e-12}).x)
        return self

    def predict(self,experiments):return predict_at(experiments,self.young_modulus)
