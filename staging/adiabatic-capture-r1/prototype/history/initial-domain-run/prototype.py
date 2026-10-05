"""Slow classical capture: static area model and direct Hamiltonian ensembles.

The trajectory reference never uses a capture fraction or final branch action.
Its initial action-to-energy inversion specifies the physical preparation only.
"""
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

@lru_cache(None)
def quadrature(n):
    return leggauss(n)

def potential(q,s,delta):
    return q*q*q*q/4-s*q*q/2+delta*s**1.5*q

@lru_cache(None)
def geometry(delta):
    roots=np.roots([1.,0.,-1.,delta])
    assert max(abs(roots.imag))<1e-10
    x=np.sort(roots.real)
    saddle=x[1];barrier=potential(saddle,1.,delta)
    b=np.sqrt(2*(1-saddle*saddle))
    lo,hi=-saddle-b,-saddle+b
    z,w=quadrature(128)
    def lobe(a,b):
        theta=np.pi*z/2;q=(a+b)/2+(b-a)*np.sin(theta)/2
        p=abs(q-saddle)*np.sqrt(np.maximum(0.,(hi-q)*(q-lo)))/np.sqrt(2)
        jac=(b-a)*np.cos(theta)*np.pi/4
        return float(2*np.sum(w*p*jac))
    areas=np.array([lobe(lo,saddle),lobe(saddle,hi)])
    return x,barrier,areas

def phase_area(energy,delta,branch=None,order=96):
    roots=np.roots([.25,0.,-.5,delta,-energy])
    real=np.sort(roots.real[abs(roots.imag)<1e-8])
    if len(real)<2:return 0.
    saddle=geometry(delta)[0][1]
    z,w=quadrature(order);theta=np.pi*z/2;total=0.
    for lo,hi in zip(real[::2],real[1::2]):
        if branch is not None and ((hi<saddle)!=(branch==0)):continue
        q=(hi+lo)/2+(hi-lo)*np.sin(theta)/2
        jac=(hi-lo)*np.cos(theta)*np.pi/4
        momentum=np.sqrt(np.maximum(0.,2*(energy-potential(q,1.,delta))))
        total+=2*np.sum(w*jac*momentum)
    return float(total)

def energy_at_area(area,delta,branch=None):
    x,barrier,_=geometry(delta)
    low=min(potential(x[[0,2]],1.,delta)) if branch is None else potential(x[0 if branch==0 else 2],1.,delta)
    lo=low+1e-11
    hi=barrier-1e-10 if branch is not None else max(1.,barrier+1.)
    if branch is not None:assert area<phase_area(hi,delta,branch)
    while phase_area(hi,delta,branch)<area:hi=2*hi+1
    return float(brentq(lambda e:phase_area(e,delta,branch)-area,lo,hi,xtol=2e-12))

def predictions(j0,delta,s_final,s_initial=.05,action_order=32):
    z,w=quadrature(action_order);actions=j0*(1.1+.1*z);weights=w/2
    _,_,areas=geometry(delta);prob=areas/areas.sum();sep=areas.sum()*s_final**1.5/(2*np.pi)
    actual=[];source=[];captured=[]
    assert actions.min()>areas.sum()*s_initial**1.5/(2*np.pi)
    for action in actions:
        area=2*np.pi*action/s_final**1.5
        global_e=s_final*s_final*energy_at_area(area,delta)
        source.append(global_e)
        if action>=sep:
            actual.append(global_e);captured.append(0.)
        else:
            energies=np.array([s_final*s_final*energy_at_area(prob[i]*area,delta,i) for i in [0,1]])
            actual.append(float(prob@energies));captured.append(1.)
    return {'physical':float(weights@actual),'source':float(weights@source),
            'capture_fraction':float(weights@captured),'right_capture_probability':float(prob[1]),
            'relative_gap':abs(float(weights@source)/float(weights@actual)-1),
            'initial_separatrix_action':float(areas.sum()*s_initial**1.5/(2*np.pi)),
            'final_separatrix_action':float(sep)}

@lru_cache(maxsize=128)
def initial_ensemble(j0,delta,s_initial,action_order,phase_order,phase_offset=.5):
    """Time-uniform orbit points at each Gauss action node, before any ramp."""
    z,w=quadrature(action_order);actions=j0*(1.1+.1*z);weights=np.repeat(w/(2*phase_order),phase_order)
    qlist=[];plist=[];initial=[];errors=[];closure=[]
    for action in actions:
        energy=s_initial*s_initial*energy_at_area(2*np.pi*action/s_initial**1.5,delta)
        turning=np.roots([.25,0.,-s_initial/2,delta*s_initial**1.5,-energy])
        roots=np.sort(turning.real[abs(turning.imag)<1e-8]);assert len(roots)==2
        left,right=roots
        # Initial period is computed from the frozen physical Hamiltonian.
        zz,ww=quadrature(192);theta=np.pi*zz/2
        q=(right+left)/2+(right-left)*np.sin(theta)/2
        jac=(right-left)*np.cos(theta)*np.pi/4
        p=np.sqrt(np.maximum(1e-300,2*(energy-potential(q,s_initial,delta))))
        period=float(2*np.sum(ww*jac/p))
        def rhs(t,y):return [y[1],-y[0]**3+s_initial*y[0]-delta*s_initial**1.5]
        orbit=solve_ivp(rhs,(0.,period),[right,0.],method='DOP853',rtol=2e-12,atol=2e-13,dense_output=True)
        assert orbit.success
        times=period*(np.arange(phase_order)+phase_offset)/phase_order
        qq,pp=orbit.sol(times)
        qlist.extend(qq);plist.extend(pp);initial.extend(np.full(phase_order,energy))
        errors.append(float(np.max(abs(pp*pp/2+potential(qq,s_initial,delta)-energy))))
        closure.append(float(np.max(abs(orbit.y[:,-1]-[right,0.]))))
    return np.array(qlist),np.array(plist),weights,np.array(initial),max(errors),max(closure)

def ramp(u,kind):
    r=float(np.clip(u,0.,1.))
    if kind=='cubic':value=r*r*(3-2*r);deriv=6*r*(1-r)
    elif kind=='quintic':value=r**3*(10-15*r+6*r*r);deriv=30*r*r*(1-r)**2
    else:raise ValueError(kind)
    return value,deriv

def trajectory(j0=.14,delta=.2,s_final=2.,s_initial=.05,duration=256.,step=.08,
               action_order=16,phase_order=128,phase_offset=.5,shape='cubic'):
    start=time.perf_counter()
    qi,pi,weights,initial,prep_error,orbit_error=initial_ensemble(j0,delta,s_initial,action_order,phase_order,phase_offset)
    q=qi.copy();p=pi.copy();work=np.zeros_like(q)
    count=int(np.ceil(duration/step));dt=duration/count
    # Fourth-order extended-phase-space symplectic composition. A drifts q,t;
    # B kicks p and the time-conjugate momentum from V(q,t).
    c=1/(2-2**(1/3));composition=(c,1-2*c,c)
    t=0.
    for _ in range(count):
        for coefficient in composition:
            h=dt*coefficient
            q+=.5*h*p;t+=.5*h
            r,dr=ramp(t/duration,shape);s=s_initial+(s_final-s_initial)*r
            ds=(s_final-s_initial)*dr/duration
            work+=h*ds*(-q*q/2+1.5*delta*np.sqrt(s)*q)
            p+=h*(-q*q*q+s*q-delta*s**1.5)
            q+=.5*h*p;t+=.5*h
    final=p*p/2+potential(q,s_final,delta)
    barrier=geometry(delta)[1]*s_final*s_final
    saddle=geometry(delta)[0][1]*np.sqrt(s_final)
    trapped=final<barrier
    right=(q>saddle)&trapped
    assert np.isfinite(final).all() and np.isfinite(work).all()
    error=final-initial-work
    return {'parameters':{'j0':j0,'delta':delta,'s_final':s_final,'s_initial':s_initial,'duration':duration,'step':dt,'action_order':action_order,'phase_order':phase_order,'phase_offset':phase_offset,'shape':shape},
      'trajectories':len(q),'mean_energy':float(weights@final),'right_lobe_fraction':float(weights@right),'trapped_fraction':float(weights@trapped),'energy_min':float(final.min()),'energy_max':float(final.max()),
      'energy_work_balance_mean_abs':float(abs(weights@error)),'energy_work_balance_max_abs':float(max(abs(error))),
      'initial_energy_error':prep_error,'initial_orbit_closure_error':orbit_error,'seconds':time.perf_counter()-start}

if __name__=='__main__':
    target=predictions(.14,.2,2.);print('AREA',json.dumps(target),flush=True)
    out=Path(__file__).with_name('exploration.jsonl')
    runs=[dict(duration=x) for x in [64.,128.,256.,512.]]+[dict(duration=256.,step=.04),dict(duration=512.,action_order=32),dict(duration=512.,phase_order=256),dict(duration=512.,shape='quintic')]
    for config in runs:
        row=trajectory(**config);row['area_prediction']=target;row['relative_energy_disagreement']=abs(row['mean_energy']/target['physical']-1)
        with out.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
