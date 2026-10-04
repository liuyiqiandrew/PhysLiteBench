"""Author-only prototype; does not modify the canonical rotating-reservoir task."""
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import solve_continuous_lyapunov

R = np.array([[0., -1.], [1., 0.]])
B = .9


def matrices(gamma, omega, tau, kx, ky, ta, tb):
    a = np.zeros((6,6))
    a[:2,2:4] = np.eye(2)
    a[2:4,:2] = -np.diag([kx,ky])
    a[2:4,2:4] = -gamma*np.eye(2)
    a[2:4,4:] = np.eye(2)
    a[4:,:2] = B*omega*R/tau
    a[4:,2:4] = -B*np.eye(2)/tau
    a[4:,4:] = omega*R-np.eye(2)/tau
    q = np.zeros((6,6))
    q[2:4,2:4] = 2*gamma*ta*np.eye(2)
    q[4:,4:] = 2*B*tb*np.eye(2)/tau**2
    return a,q


def evaluate(gamma=.67, omega=.5, tau=.5, kx=1.2, ky=2.5, ta=1., tb=1.):
    controls=dict(gamma=gamma,omega=omega,tau=tau,kx=kx,ky=ky,ta=ta,tb=tb)
    a,q=matrices(**controls)
    stability=float(np.linalg.eigvals(a).real.max())
    assert stability<0
    c=solve_continuous_lyapunov(a,-q)
    lab=-float(np.trace(c[4:,2:4]))
    motor=float(omega*np.trace(c[4:,:2]@R.T))
    heat=lab+motor
    calorimeter=float(np.trace(c[4:,4:])/B-2*tb/tau)
    qa=float(gamma*(np.trace(c[2:4,2:4])-2*ta))
    return {'controls':controls,'heat':heat,'laboratory_energy':lab,'motor':motor,
            'auxiliary_calorimeter':calorimeter,'heat_a':qa,
            'entropy_rate':qa/ta+heat/tb,'balance_error':abs(qa+heat-motor),
            'heat_identity_error':abs(calorimeter-heat),'stability':stability,
            'covariance_min_eigenvalue':float(np.linalg.eigvalsh(c).min())},c


def frequency_covariance(controls):
    a,q=matrices(**controls)
    def spectrum(w):
        resolvent=np.linalg.inv(-1j*w*np.eye(6)-a)
        return (resolvent@q@resolvent.conj().T).real/np.pi
    covariance,error=quad_vec(spectrum,0,np.inf,epsabs=2e-10,epsrel=2e-10)
    return covariance,float(error)


report={'prototype':'rotating-reservoir Maxwell-memory bath','hidden_screen':[]}
for omega,tau,kx,ky,ta,tb in [(.55,.2,.9,2.1,1.,1.),(-.6,.7,1.8,3.2,1.,1.),
                              (.6,1.5,1.1,2.5,1.,1.),(-.65,.5,1.6,2.2,1.4,.8),
                              (.6,.8,1.3,2.8,.8,1.2),(-.5,1.2,.8,3.,1.,1.)]:
    row,c=evaluate(omega=omega,tau=tau,kx=kx,ky=ky,ta=ta,tb=tb)
    row['relative_error']=abs((row['laboratory_energy']-row['heat'])/row['heat'])
    ref,error=frequency_covariance(row['controls'])
    row['independent_frequency_covariance_error']=float(abs(ref-c).max())
    row['quadrature_error_estimate']=error
    report['hidden_screen'].append(row)

# At rest, equal-temperature baths give the extended canonical Gibbs state.
rest=[]
for gamma,tau,kx,ky,t in product([.4,.67,1.1],[.1,.5,1.5],[.8,1.8],[2.,3.2],[.6,1.4]):
    row,c=evaluate(gamma=gamma,omega=0.,tau=tau,kx=kx,ky=ky,ta=t,tb=t)
    canonical=np.diag([t/kx,t/ky,t,t,B*t/tau,B*t/tau])
    rest.append(max(float(abs(c-canonical).max()),abs(row['heat'])))
report['rest_gibbs_max_error']=max(rest)

# With A removed and an isotropic trap, the co-rotating Gibbs ensemble is exact.
rotating=[]
for omega,tau,k,t in product([-.65,.3,.65],[.1,.5,1.5],[.8,1.8],[.6,1.4]):
    row,c=evaluate(gamma=0.,omega=omega,tau=tau,kx=k,ky=k,ta=t,tb=t)
    transform=np.eye(6);transform[2:4,:2]=-omega*R
    corotating=transform@c@transform.T
    expected=np.diag([t/(k-omega**2),t/(k-omega**2),t,t,B*t/tau,B*t/tau])
    rotating.append(max(float(abs(corotating-expected).max()),abs(row['heat'])))
report['isolated_rotating_bath_gibbs_max_error']=max(rotating)

# Independent closed-form heat current at Omega=0, derived from three stationary moments.
calibration=[]
for gamma,tau,kx,ky,ta,tb in product([.4,.67,1.1],[.1,.35,.6],[.8,1.8],[2.,3.2],[.6,1.6],[.6,1.4]):
    row,c=evaluate(gamma=gamma,omega=0.,tau=tau,kx=kx,ky=ky,ta=ta,tb=tb)
    exact=sum(B*(ta-tb)/(k*tau*tau+(gamma+B)*tau+1+B/gamma) for k in [kx,ky])
    calibration.append(abs(row['heat']-exact))
report['rest_current_moment_formula_error']=max(calibration)
report['calibration_identifiability']='For any nonzero temperature contrast and tau<=.6, every modal conductance b/[k*tau^2+(gamma+b)*tau+1+b/gamma] is strictly increasing across gamma .4..1.1 because b/gamma^2>tau.'

# Frequency-domain thermal-force covariance equals the Hermitian part of the memory kernel.
fdt=[]
for omega,tau,w in product([-.65,0.,.65],[.1,.5,1.5],[0.,.3,1.,3.,10.]):
    h=np.linalg.inv(np.eye(2)-tau*omega*R-1j*w*tau*np.eye(2))
    memory=B*h
    fdt.append(float(abs(2*B*h@h.conj().T-(memory+memory.conj().T)).max()))
report['memory_fdt_max_error']=max(fdt)

corners=[]
for gamma,omega,tau,kx,ky,ta,tb in product([.4,1.1],[-.65,.65],[.1,1.5],[.8,1.8],[2.,3.2],[.6,1.6],[.6,1.4]):
    row,_=evaluate(gamma=gamma,omega=omega,tau=tau,kx=kx,ky=ky,ta=ta,tb=tb)
    corners.append(row)
rng=np.random.default_rng(728231)
for _ in range(1000):
    row,_=evaluate(gamma=rng.uniform(.4,1.1),omega=rng.uniform(-.65,.65),tau=rng.uniform(.1,1.5),
                   kx=rng.uniform(.8,1.8),ky=rng.uniform(2,3.2),ta=rng.uniform(.6,1.6),tb=rng.uniform(.6,1.4))
    corners.append(row)
report['supported_domain']={'cases':len(corners),'max_stability':max(r['stability'] for r in corners),
    'minimum_covariance_eigenvalue':min(r['covariance_min_eigenvalue'] for r in corners),
    'minimum_entropy_rate':min(r['entropy_rate'] for r in corners),
    'max_energy_balance_error':max(r['balance_error'] for r in corners),
    'max_auxiliary_heat_identity_error':max(r['heat_identity_error'] for r in corners)}
assert report['supported_domain']['minimum_entropy_rate']>=-1e-10
assert report['supported_domain']['max_energy_balance_error']<1e-11
assert report['supported_domain']['max_auxiliary_heat_identity_error']<1e-10

# In the short-memory limit the full4D rotating Markov result is recovered.
limits=[]
for controls in [dict(gamma=.67,omega=.6,kx=.9,ky=2.2,ta=1.2,tb=.8),
                 dict(gamma=.9,omega=-.5,kx=1.6,ky=2.8,ta=.8,tb=1.3)]:
    g,w,kx,ky,ta,tb=[controls[x] for x in ['gamma','omega','kx','ky','ta','tb']]
    a=np.block([[np.zeros((2,2)),np.eye(2)],[-np.diag([kx,ky])+B*w*R,-(g+B)*np.eye(2)]])
    q=np.diag([0.,0.,2*(g*ta+B*tb),2*(g*ta+B*tb)])
    c0=solve_continuous_lyapunov(a,-q)
    transform=np.column_stack([-w*R,np.eye(2)])
    heat0=float(B*np.trace(transform@c0@transform.T)-2*B*tb)
    entries=[]
    for tau in [1e-2,1e-3,1e-4,1e-5]:
        row,c=evaluate(**controls,tau=tau)
        entries.append({'tau':tau,'heat_error':abs(row['heat']-heat0),
                        'position_velocity_covariance_error':float(abs(c[:4,:4]-c0).max())})
    limits.append({'controls':controls,'markov_heat':heat0,'refinement':entries})
report['white_memory_limits']=limits
assert report['rest_gibbs_max_error']<1e-10
assert report['isolated_rotating_bath_gibbs_max_error']<1e-10
assert report['rest_current_moment_formula_error']<1e-10
path=Path(__file__).resolve().parents[1]/'results/rotating-reservoir-memory-prototype.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
