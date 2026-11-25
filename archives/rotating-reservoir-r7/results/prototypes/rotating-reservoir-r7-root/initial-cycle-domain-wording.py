"""Author proposal only: finite heat bias versus a Gaussian long-time closure.

Independent exact renewal integral for zero-flow Maxwell collisions, plus a
positive grid operator gauged by the reservoir-frame boundary kinetic energy.
Neither this prototype nor its runs are model evaluations.
"""
from pathlib import Path
import importlib.util,json,time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.linalg import eigvals

STAGE=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('frozen_rotor_r6',STAGE/'tasks/rotating-reservoir/solution/model.py')
r6=importlib.util.module_from_spec(spec);spec.loader.exec_module(r6)

def renewal(chi,ta,tb,gamma=.67,rate=3.6):
    coefficient=chi*(ta-tb)-chi*chi*ta*tb
    assert coefficient>-1
    def residual(theta):
        return rate*quad(lambda t:np.exp(-(rate+theta)*t)/np.sqrt(1+coefficient*(-np.expm1(-2*gamma*t))),0,np.inf,epsabs=2e-12,epsrel=2e-12)[0]-1
    lo=-rate+1e-4;hi=1.
    while residual(hi)>0:hi*=2
    return float(brentq(residual,lo,hi,xtol=2e-13))

def bernoulli(z):
    z=np.asarray(z);small=abs(z)<1e-5
    safe=np.where(small,1.,z)
    return np.where(small,1-z/2+z*z/12-z**4/720,safe/np.expm1(safe))

def grid(chi,ta,tb,omega=0.,eta=0.,flip=1.,gamma=.67,points=241,extent=9.):
    width=extent*np.sqrt(max(ta,tb))+abs(omega)
    h=2*width/points;x=-width+h*(np.arange(points)+.5);faces=(x[:-1]+x[1:])/2
    diffusion=gamma*ta; drift=-gamma*faces
    up=diffusion/h**2*bernoulli(-drift*h/diffusion)
    down=diffusion/h**2*bernoulli(drift*h/diffusion)
    continuous=np.diag(up,1)+np.diag(down,-1)
    continuous-=np.diag(continuous.sum(axis=1))
    blocks=[];gauges=[]
    for sign in (1.,-1.):
        flow=sign*omega;energy=(x-flow)**2/2
        logd=-chi*energy
        # D^-1 L_chi D: the collision mark cancels, while the diffusion
        # and valve jumps receive the endpoint-energy similarity factors.
        gauged_continuous=continuous*np.exp(logd[None,:]-logd[:,None])
        density=np.exp(-(x-flow)**2/(2*tb))/np.sqrt(2*np.pi*tb)*h
        collision=3.6*((1-eta)+eta*abs(x[:,None]-x[None,:]))*density[None,:]
        block=gauged_continuous+collision-np.diag(collision.sum(axis=1))-flip*np.eye(points)
        blocks.append(block);gauges.append(logd)
    off0=np.diag(flip*np.exp(gauges[1]-gauges[0]))
    off1=np.diag(flip*np.exp(gauges[0]-gauges[1]))
    matrix=np.block([[blocks[0],off0],[off1,blocks[1]]])
    values=eigvals(matrix,check_finite=False)
    principal=values[np.argmax(values.real)]
    assert abs(principal.imag)<1e-8
    return float(principal.real)

def extrapolated(*args,points=161,**kwargs):
    coarse=grid(*args,points=points,**kwargs)
    fine=grid(*args,points=2*points,**kwargs)
    return (4*fine-coarse)/3

def run():
    start=time.perf_counter();rows=[]
    for ta,tb,omega,eta,chi in [(1.4,1.4,0.,0.,.35),(1.4,1.4,0.,1.,.35),
        (.6,1.4,0.,1.,.35),(1.6,.6,0.,1.,-.35),(1.,1.,.65,1.,.35),
        (.6,1.4,-.65,1.,.35),(1.6,.6,.65,1.,-.35),(1.4,1.4,.65,1.,-.35)]:
        e=dict(temperature_a=ta,temperature_b=tb,angular_speed=omega,flux_fraction=eta,flip_rate=1.)
        moments=r6.physical_statistics(e,.67)
        gaussian=-chi*moments['mean']+chi*chi*moments['noise']/2
        full=extrapolated(chi,ta,tb,omega,eta)
        row=dict(input=dict(e,heat_bias=chi),scgf=full,gaussian=gaussian,
                 relative_error=abs(full-gaussian)/abs(full),
                 moments={k:moments[k] for k in ['mean','noise']})
        if eta==0 and omega==0:
            exact=renewal(chi,ta,tb)
            row.update(exact_renewal=exact,grid_renewal_error=abs(full-exact))
        rows.append(row);print(json.dumps(row),flush=True)
    anchors=[]
    for ta,tb in [(1.4,1.4),(.6,1.4),(1.6,.6)]:
        chi=.2
        value=renewal(chi,ta,tb)
        gc=renewal(1/tb-1/ta-chi,ta,tb)
        delta=1e-4
        mean=-(renewal(delta,ta,tb)-renewal(-delta,ta,tb))/(2*delta)
        noise=(renewal(delta,ta,tb)+renewal(-delta,ta,tb))/delta**2
        exact=r6.maxwell_statistics(dict(temperature_a=ta,temperature_b=tb,angular_speed=0.,flip_rate=1.),.67,3.6)
        anchors.append(dict(ta=ta,tb=tb,gc_error=abs(value-gc),mean_error=abs(mean-exact['mean']),noise_error=abs(noise-exact['noise'])))
    result=dict(cases=rows,anchors=anchors,seconds=time.perf_counter()-start,
       domain='Tilted heat exponential is finite in -1/Ta < chi < 1/Tb; |chi|<=.35 lies strictly inside every stated temperature case.',
       physical_distinction='Finite exponential bias probes correlated non-Gaussian heat trajectories even after exact mean and variance are retained.')
    target=Path(__file__).with_suffix('.json');target.write_text(json.dumps(result,indent=2)+'\n')
    print('saved',target,flush=True)

if __name__=='__main__':run()
