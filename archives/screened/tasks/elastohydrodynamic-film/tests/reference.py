"""Bulk strain-energy minimization and direct conservative pressure propagator."""
from functools import lru_cache
import json
import numpy as np
from scipy.linalg import cho_factor,cho_solve,expm
from numpy.polynomial.legendre import leggauss,Legendre
PARAMETER='young_modulus'
TRUE_PARAMETER=1.1


def experiment(q=.9,contrast=.6,pattern=1,pattern_phase=.4,initial=1,initial_phase=0.,detector=1,detector_phase=0.,time=.2,thickness=1.,gap=10.,pressure=50.):
    return dict(fundamental_wavenumber=float(q),contrast=float(contrast),pattern_mode=int(pattern),pattern_phase=float(pattern_phase),pressure_mode=int(initial),pressure_phase=float(initial_phase),detector_mode=int(detector),detector_phase=float(detector_phase),elapsed_time=float(time),thickness=float(thickness),gap=float(gap),pressure=float(pressure))


def calibration_inputs():
    return [experiment(q,0.,initial=m,detector=m,time=t,thickness=d,gap=h)
            for q in [.6,.9,1.2] for m in [1,2] for t in [0.,.1,.3] for d in [.8,1.2] for h in [8.,12.]]


def hidden_inputs():
    return {'mixed_harmonics':[experiment(q,a,1,.4,1,0.,m,.3,t) for q in [.8,1.2] for a in [.6,.7] for t in [.1,.4] for m in [0,1,2,3]],
            'pattern_spacing':[experiment(q,.65,2,.3,2,.2,m,.1,t) for q in [.6,1.1] for t in [.05,.2,.5] for m in [0,2,4]],
            'phase_rotation':[experiment(1.2,.7,1,phase,2,.2,m,.2,t,thickness=1.2) for phase in [-.6,1.2] for t in [.1,.35] for m in [1,2,3,4]]}


@lru_cache(None)
def grid(points):
    x=np.arange(points)*2*np.pi/points;modes=np.fft.fftfreq(points,1/points)
    derivative=np.fft.ifft(1j*modes[:,None]*np.fft.fft(np.eye(points),axis=0),axis=0).real
    return x,derivative


@lru_cache(128)
def elastic_problem(q,contrast,pattern,phase,points=33,degree=16):
    x,d=grid(points);d=q*d;r=np.diag(1+contrast*np.cos(pattern*x+phase))
    z,w=leggauss(2*degree+8);s=(z+1)/2;w=w/2
    basis=np.array([s*Legendre.basis(j)(z) for j in range(degree)]).T
    gradient=np.array([Legendre.basis(j)(z)+2*s*Legendre.basis(j).deriv()(z) for j in range(degree)]).T
    bb=basis.T@(w[:,None]*basis);gg=gradient.T@(w[:,None]*gradient);bg=basis.T@(w[:,None]*gradient)
    mu=1/(2*(1+.48));lam=.48/((1+.48)*(1-2*.48));longitudinal=lam+2*mu
    uu=longitudinal*np.kron(bb,d.T@r@d)+mu*np.kron(gg,r)
    ww=longitudinal*np.kron(gg,r)+mu*np.kron(bb,d.T@r@d)
    uw=lam*np.kron(bg,d.T@r)+mu*np.kron(bg.T,r@d)
    stiffness=np.block([[uu,uw],[uw.T,ww]])
    surface=np.concatenate([np.zeros((points,points*degree)),np.tile(np.eye(points),(1,degree))],axis=1)
    return stiffness,surface


@lru_cache(128)
def elastic_compliance(q,contrast,pattern,phase,points=33,degree=16):
    stiffness,surface=elastic_problem(q,contrast,pattern,phase,points,degree)
    compliance=surface@cho_solve(cho_factor(stiffness),surface.T)
    return (compliance+compliance.T)/2


@lru_cache(256)
def fields(key,young_modulus,points=33,degree=16):
    e=json.loads(key);x,d=grid(points);depth=e['thickness']*.001;k=e['fundamental_wavenumber']*1000
    compliance=depth/(young_modulus*1e6)*elastic_compliance(k*depth,e['contrast'],e['pattern_mode'],e['pattern_phase'],points,degree)
    # Surface gap volume is the conserved field. Its restoring pressure is
    # obtained from the variational elastic response; no pressure-mean pin.
    mobility=(e['gap']*1e-6)**3/(12*.15)
    flow=-mobility*k*k*d@d
    initial=e['pressure']*np.cos(e['pressure_mode']*x+e['pressure_phase'])
    pressure=expm(-np.linalg.solve(compliance,flow)*e['elapsed_time'])@initial
    return x,compliance@pressure


def predict(experiments,young_modulus,points=33,degree=16):
    out=[]
    for e in experiments:
        x,displacement=fields(json.dumps(e,sort_keys=True),young_modulus,points,degree)
        if e['detector_mode']==0:value=np.mean(displacement)
        else:value=2*np.mean(displacement*np.cos(e['detector_mode']*x+e['detector_phase']))
        out.append(float(value*1e9))
    return np.array(out)
