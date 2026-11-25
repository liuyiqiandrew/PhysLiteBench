"""Bounded compliant-contact prototype. No task or model evaluations."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time
from pathlib import Path
from itertools import product
import numpy as np
from scipy.linalg import solve_continuous_lyapunov,expm
from scipy.integrate import quad_vec
R=np.array([[0.,-1.],[1.,0.]])
B=.9


def matrices(gamma,omega,strain,tau,kx,ky,ta,tb):
    A=omega*R+np.diag([strain,-strain]);K=np.diag([kx,ky])
    M=np.zeros((6,6));M[:2,2:4]=np.eye(2)
    M[2:4,:2]=-K;M[2:4,2:4]=-gamma*np.eye(2);M[2:4,4:]=np.eye(2)
    M[4:,:2]=B*A/tau;M[4:,2:4]=-B*np.eye(2)/tau;M[4:,4:]=A-np.eye(2)/tau
    Q=np.diag([0.,0.,2*gamma*ta,2*gamma*ta,2*B*tb/tau**2,2*B*tb/tau**2])
    return M,Q,A


def evaluate(gamma=.67,omega=.3,strain=.25,tau=.5,kx=1.2,ky=2.5,ta=1.,tb=1.):
    e=dict(gamma=gamma,omega=omega,strain=strain,tau=tau,kx=kx,ky=ky,ta=ta,tb=tb)
    M,Q,A=matrices(**e);stability=float(np.max(np.linalg.eigvals(M).real))
    if stability>=0: return {'controls':e,'stable':False,'max_drift_real_eigenvalue':stability}
    C=solve_continuous_lyapunov(M,-Q)
    # z is the spring force on the inertial particle; q=r+tau*z/b.
    zw=float(np.trace(C[4:,2:4])-np.trace(C[4:,:2]@A.T))
    source=-zw
    stress_power=float(tau/B*np.trace(C[4:,4:]@A.T))
    heat=float(np.trace(C[4:,4:])/B-2*tb/tau)
    qa=float(gamma*(np.trace(C[2:4,2:4])-2*ta))
    motor=float(np.trace(C[4:,:2]@A.T)+stress_power)
    return {'controls':e,'stable':True,'heat':heat,'relative_particle_work':source,'elastic_flow_power':stress_power,
            'heat_a':qa,'flow_power':motor,'absolute_gap':abs(heat-source),'relative_gap':abs(heat-source)/max(abs(heat),1e-30),
            'heat_identity_error':abs(heat-source-stress_power),'total_energy_error':abs(qa+heat-motor),
            'entropy_rate':qa/ta+heat/tb,'max_drift_real_eigenvalue':stability,
            'minimum_covariance_eigenvalue':float(np.linalg.eigvalsh(C).min()),'covariance':C.tolist()}


def reference(e):
    # Independently assemble the actual particle and bath bead coordinates (r,v,q).
    g,w,s,t,kx,ky,ta,tb=[e[k] for k in ['gamma','omega','strain','tau','kx','ky','ta','tb']]
    h=B/t;A=np.array([[s,-w],[w,-s]])
    M=np.zeros((6,6));M[:2,2:4]=np.eye(2)
    M[2:4,:2]=-np.diag([kx,ky])-h*np.eye(2);M[2:4,2:4]=-g*np.eye(2);M[2:4,4:]=h*np.eye(2)
    M[4:,:2]=np.eye(2)/t;M[4:,4:]=A-np.eye(2)/t
    noise=np.zeros((6,4));noise[2:4,:2]=np.sqrt(2*g*ta)*np.eye(2);noise[4:,2:]=np.sqrt(2*tb/B)*np.eye(2)
    def impulse(time):
        G=expm(M*time)@noise
        return G@G.T
    C,error=quad_vec(impulse,0,np.inf,epsabs=2e-11,epsrel=2e-11)
    extension=np.column_stack([np.eye(2),np.zeros((2,2)),-np.eye(2)])
    heat=float(h*h/B*np.trace(extension@C@extension.T)-2*h*tb/B)
    transform=np.eye(6);transform[4:,:2]=-h*np.eye(2);transform[4:,4:]=h*np.eye(2)
    return heat,transform@C@transform.T,float(error)


def main():
    start=time.monotonic();rows=[]
    # Keep every screened case, including zero, weak and sign-changing signals.
    for w,s,t,ta,tb in product([0.,.3,.6],[-.4,-.2,0.,.2,.4],[.2,.6,1.0],[(1.)],[(1.)]):
        rows.append(evaluate(omega=w,strain=s,tau=t,ta=ta,tb=tb))
    diagnostics=[]
    for g,w,s,t,kx,ky,ta,tb in [(.4,.3,.4,.6,1.2,2.5,1.,1.),(.67,.3,.4,.6,1.2,2.5,1.,1.),(1.1,.3,.4,.6,1.2,2.5,1.,1.),
       (.67,-.45,-.35,.8,1.5,2.4,1.2,.9),(.67,.2,.3,.7,1.,2.8,.9,1.1),(.67,.55,-.3,.9,1.4,2.2,1.4,.8)]:
        r=evaluate(g,w,s,t,kx,ky,ta,tb);assert r['stable'];h,c,err=reference(r['controls'])
        r.update(reference_heat=h,reference_heat_error=abs(h-r['heat']),reference_covariance_error=float(abs(c-np.array(r['covariance'])).max()),reference_quadrature_error=err)
        diagnostics.append(r)
    calibration=[]
    for g in np.linspace(.4,1.1,29):
        for w,t,ta,tb in [(0.,.2,1.4,.8),(0.,.6,.8,1.3),(.4,.3,1.3,.8),(-.55,.5,1.1,.9)]:
            r=evaluate(gamma=float(g),omega=w,strain=0.,tau=t,ta=ta,tb=tb)
            if w==0:
                exact=sum(B*(ta-tb)/(k*t*t+(g+B)*t+1+B/g) for k in [1.2,2.5])
                derivative=sum(B*(ta-tb)*(B/g**2-t)/(k*t*t+(g+B)*t+1+B/g)**2 for k in [1.2,2.5])
                r.update(rest_analytic_error=abs(exact-r['heat']),rest_derivative=float(derivative))
            calibration.append(r)
    corners=[]
    for g,w,s,t,kx,ky,ta,tb in product([.4,1.1],[-.6,.6],[-.4,.4],[.2,.8],[1.,1.6],[2.2,2.8],[.8,1.4],[.8,1.4]):
        corners.append(evaluate(g,w,s,t,kx,ky,ta,tb))
    stable=[r for r in rows+diagnostics+calibration+corners if r['stable']]
    report={'status':'bounded hardening prototype; no task or model evaluation','screen':rows,'diagnostics':diagnostics,'calibration':calibration,'corners':corners,
      'summary':{'screen_count':len(rows),'diagnostic_count':len(diagnostics),'calibration_count':len(calibration),'corner_count':len(corners),
       'unstable_corners':sum(not r['stable'] for r in corners),'max_calibration_difference':max(r['absolute_gap'] for r in calibration),
       'max_reference_heat_error':max(r['reference_heat_error'] for r in diagnostics),'max_reference_covariance_error':max(r['reference_covariance_error'] for r in diagnostics),
       'max_heat_identity_error':max(r['heat_identity_error'] for r in stable),'max_energy_error':max(r['total_energy_error'] for r in stable),
       'minimum_entropy_rate':min(r['entropy_rate'] for r in stable),'run_seconds':time.monotonic()-start}}
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))
    for r in diagnostics:print('DIAGNOSTIC',r['heat'],r['relative_particle_work'],r['relative_gap'],r['controls'])
    assert report['summary']['max_calibration_difference']<1e-10
    assert report['summary']['max_reference_heat_error']<1e-9
    assert report['summary']['max_heat_identity_error']<1e-9
    assert report['summary']['max_energy_error']<1e-9

if __name__=='__main__': main()
