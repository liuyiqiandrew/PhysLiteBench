"""Independent elastic energy minimization and conservative gap evolution."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss,Legendre
from scipy.integrate import solve_ivp

PARAMETER='young_modulus'
TRUE_PARAMETER=1.1


def experiment(wavenumber=1.,thickness=1.,gap=10.,elapsed_time=.2,pressure=60.):
    return dict(wavenumber=float(wavenumber),thickness=float(thickness),gap=float(gap),
                elapsed_time=float(elapsed_time),pressure=float(pressure))


def calibration_inputs():
    return [experiment(0.,d,5+i%11,.1*(i%7),p)
            for i,(d,p) in enumerate(zip(np.linspace(.5,1.5,120),25+70*np.sin(np.arange(120)*.37)**2))]


def hidden_inputs():
    return {'pressure_waves':[experiment(k,1.,10.,t) for k in [.5,1.,2.,3.] for t in [.02,.08,.25,.7]],
            'layer_depth':[experiment(k,d,8.,t) for d in [.6,1.,1.4] for k in [.7,1.5,2.5] for t in [.05,.2,.6]],
            'film_mobility':[experiment(1.3,1.2,h,t) for h in [6.,10.,14.] for t in [.02,.1,.4,.9]]}


@lru_cache(None)
def elastic_energy_compliance(q,basis_count=18):
    # Dimensionless layer -1<z<0, unit Young modulus. Displacement amplitudes
    # are ux=U(z)sin(kx), uz=W(z)cos(kx). Every basis function vanishes at
    # the bonded bottom. Minimize bulk strain energy minus surface work.
    x,w=leggauss(2*basis_count+8);s=(x+1)/2;w=w/2
    basis=np.array([s*Legendre.basis(j)(x) for j in range(basis_count)]).T
    derivative=np.array([Legendre.basis(j)(x)+2*s*Legendre.basis(j).deriv()(x) for j in range(basis_count)]).T
    zero=np.zeros_like(basis)
    exx=np.concatenate([q*basis,zero],axis=1)
    ezz=np.concatenate([zero,derivative],axis=1)
    shear=np.concatenate([derivative,-q*basis],axis=1)
    trace=exx+ezz
    mu=1/(2*(1+.48));lam=.48/((1+.48)*(1-2*.48))
    stiffness=(lam*trace.T@(w[:,None]*trace)+2*mu*exx.T@(w[:,None]*exx)
               +2*mu*ezz.T@(w[:,None]*ezz)+mu*shear.T@(w[:,None]*shear))
    load=np.r_[np.zeros(basis_count),-np.ones(basis_count)]
    displacement=np.linalg.solve(stiffness,load)
    compliance=-np.sum(displacement[basis_count:])
    return float(compliance)


def predict(experiments,young_modulus,basis_count=18):
    out=[]
    for e in experiments:
        depth=e['thickness']*.001;k=e['wavenumber']*1000
        compliance=depth/(young_modulus*1e6)*elastic_energy_compliance(k*depth,basis_count)
        initial_gap_change=e['pressure']*compliance
        # The gap is the conserved liquid volume per area. Pressure is its
        # variational elastic restoring force; axial flux is Poiseuille flow.
        mobility=(e['gap']*1e-6)**3/(12*.15)
        time=e['elapsed_time']
        if time==0 or k==0:final=initial_gap_change
        else:
            def balance(t,height):return -mobility*k*k*height/compliance
            result=solve_ivp(balance,(0,time),[initial_gap_change],rtol=2e-11,atol=1e-20)
            final=result.y[0,-1]
        out.append(float(final*1e9))
    return np.array(out)
