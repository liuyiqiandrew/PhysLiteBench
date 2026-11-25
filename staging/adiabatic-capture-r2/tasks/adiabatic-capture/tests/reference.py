from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq


@lru_cache(None)
def quadrature(order):
    return leggauss(order)


def potential(q, delta):
    return q**4/4 - q*q/2 + delta*q


def phase_volume(energy, delta, order=96):
    roots = np.roots([.25, 0., -.5, delta, -energy])
    roots = np.sort(roots.real[abs(roots.imag) < 1e-8])
    if len(roots) < 2:
        return 0.
    z, w = quadrature(order)
    theta = np.pi*z/2
    total = 0.
    for left, right in zip(roots[::2], roots[1::2]):
        q = (right+left)/2 + (right-left)*np.sin(theta)/2
        jacobian = (right-left)*np.cos(theta)*np.pi/4
        momentum = np.sqrt(np.maximum(0., 2*(energy-potential(q, delta))))
        total += 2*np.sum(w*jacobian*momentum)
    return float(total)


def energy_for_volume(volume, delta):
    extrema = np.sort(np.roots([1., 0., -1., delta]).real)
    lower = float(min(potential(extrema[[0, 2]], delta))) + 1e-12
    upper = 1.
    while phase_volume(upper, delta) < volume:
        upper = 2*upper+1
    return float(brentq(lambda energy: phase_volume(energy, delta)-volume,
                        lower, upper, xtol=2e-12))


from scipy.integrate import solve_ivp
import time

TRUE_PARAMETER = .1437


def calibration_inputs():
    return [{'delta_final': .2, 'path_slope': 0., 's_final': 1.8}]


def hidden_inputs():
    return {
        'constant_path_anchor': [{'delta_final': .15, 'path_slope': 0., 's_final': 1.8},
                                 {'delta_final': .25, 'path_slope': 0., 's_final': 2.2}],
        'lower_tilt': [{'delta_final': .15, 'path_slope': .08, 's_final': 1.8},
                       {'delta_final': .15, 'path_slope': .10, 's_final': 2.2}],
        'middle_tilt': [{'delta_final': .20, 'path_slope': .10, 's_final': 1.8},
                        {'delta_final': .20, 'path_slope': .12, 's_final': 2.2}],
        'upper_tilt': [{'delta_final': .25, 'path_slope': .10, 's_final': 1.8},
                       {'delta_final': .25, 'path_slope': .12, 's_final': 2.2}],
    }


def hamiltonian(q,p,s,delta):
    return p*p/2+q**4/4-s*q*q/2+delta*s**1.5*q


@lru_cache(maxsize=128)
def initial_ensemble(action_scale,delta,action_order,phase_order,phase_offset):
    s=.05; z,w=quadrature(action_order)
    qlist=[];plist=[];elist=[];prep_error=closure=0.
    for j in action_scale*(1.1+.1*z):
        energy=s*s*energy_for_volume(2*np.pi*j/s**1.5,delta)
        turning=np.roots([.25,0.,-s/2,delta*s**1.5,-energy])
        roots=np.sort(turning.real[abs(turning.imag)<1e-8])
        if len(roots)!=2:
            raise ValueError('Initial action requires a connected orbit.')
        left,right=roots
        zz,ww=quadrature(192);theta=np.pi*zz/2
        q=(right+left)/2+(right-left)*np.sin(theta)/2
        jacobian=(right-left)*np.cos(theta)*np.pi/4
        speed=np.sqrt(np.maximum(1e-300,2*(energy-hamiltonian(q,0.,s,delta))))
        period=float(2*np.sum(ww*jacobian/speed))
        def rhs(t,y):
            return [y[1],-y[0]**3+s*y[0]-delta*s**1.5]
        orbit=solve_ivp(rhs,(0.,period),[right,0.],method='DOP853',
                        rtol=2e-12,atol=2e-13,dense_output=True)
        if not orbit.success:
            raise RuntimeError(orbit.message)
        qq,pp=orbit.sol(period*(np.arange(phase_order)+phase_offset)/phase_order)
        qlist.extend(qq);plist.extend(pp);elist.extend(np.full(phase_order,energy))
        prep_error=max(prep_error,float(np.max(abs(hamiltonian(qq,pp,s,delta)-energy))))
        closure=max(closure,float(np.max(abs(orbit.y[:,-1]-[right,0.]))))
    weights=np.repeat(w/(2*phase_order),phase_order)
    return np.asarray(qlist),np.asarray(plist),weights,np.asarray(elist),prep_error,closure


@lru_cache(maxsize=256)
def trajectory(action_scale,delta_final,path_slope,s_final,duration=512.,step=.04,
               action_order=32,phase_order=256,phase_offset=.5):
    start=time.perf_counter()
    delta_initial=delta_final+path_slope*np.log(.05/s_final)
    qi,pi,weights,initial_energy,prep_error,closure=initial_ensemble(
        action_scale,float(delta_initial),action_order,phase_order,phase_offset)
    q=qi.copy();p=pi.copy();work=np.zeros_like(q)
    count=int(np.ceil(duration/step));dt=duration/count;t=0.
    c=1/(2-2**(1/3));composition=(c,1-2*c,c)
    for _ in range(count):
        for coefficient in composition:
            h=dt*coefficient
            q+=.5*h*p;t+=.5*h
            r=float(np.clip(t/duration,0.,1.))
            s=.05+(s_final-.05)*r*r*(3-2*r)
            ds=(s_final-.05)*6*r*(1-r)/duration
            delta=delta_final+path_slope*np.log(s/s_final)
            work+=h*ds*(-q*q/2+(1.5*delta+path_slope)*np.sqrt(s)*q)
            p+=h*(-q*q*q+s*q-delta*s**1.5)
            q+=.5*h*p;t+=.5*h
    final=hamiltonian(q,p,s_final,delta_final)
    result={'mean_energy':float(weights@final),
            'energy_work_max_abs':float(np.max(abs(final-initial_energy-work))),
            'initial_energy_max_abs':prep_error,'orbit_closure_max_abs':closure,
            'seconds':time.perf_counter()-start,'trajectory_count':len(q)}
    return result



def predict(experiments, action_scale, **configuration):
    return np.asarray([trajectory(float(action_scale), float(e['delta_final']),
                                  float(e['path_slope']), float(e['s_final']),
                                  **configuration)['mean_energy'] for e in experiments])
