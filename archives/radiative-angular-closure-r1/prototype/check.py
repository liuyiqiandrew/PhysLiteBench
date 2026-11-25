"""Unevaluated weak-response M1 closure prototype."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm


def pressure(state):
    E=state[0];F=np.asarray(state[1:]);r=np.linalg.norm(F)
    if r==0:return np.eye(3)*E/3
    f=r/E;n=F/r;s=np.sqrt(4-3*f*f);den=5+2*s
    b=f*f*(6+3/(2+s))/den;a=(1-b)/3
    return E*(a*np.eye(3)+b*np.outer(n,n))


def flux_jacobian(state):
    E=state[0];F=np.asarray(state[1:]);r=np.linalg.norm(F)
    A=np.zeros((4,4));A[0,1]=1
    if r==0:
        A[1,0]=1/3
        return A
    f=r/E;n=F/r;s=np.sqrt(4-3*f*f);den=5+2*s
    b=f*f*(6+3/(2+s))/den;a=(1-b)/3
    cp=(8*f*den+6*f*(3+4*f*f)/s)/den**2
    nn=np.outer(n,n);identity=np.eye(3)
    A[1:,0]=((a+f*cp/2)*identity+(b-3*f*cp/2)*nn)[0,:]
    for j in range(3):
        direction=identity[:,j]
        derivative=cp*n[j]*(3*nn-identity)/2+(f*(6+3/(2+s))/den)*(np.outer(direction,n)+np.outer(n,direction)-2*n[j]*nn)
        A[1:,1+j]=derivative[0,:]
    return A


def predict(directions,weights,modulations,k,t,x,kappa=.28):
    n=np.array(directions,dtype=float);w=np.array(weights);a=np.array(modulations)
    state=np.r_[sum(w),w@n];delta=np.r_[sum(w*a),(w*a)@n]
    A=flux_jacobian(state)
    surrogate=np.real((expm(-1j*k*t*A)@delta)[0]*np.exp(1j*k*x))*np.exp(-kappa*t)
    exact=np.sum(w*a*np.cos(k*(x-n[:,0]*t)))*np.exp(-kappa*t)
    return float(exact),float(surrogate)


def main():
    rng=np.random.default_rng(103157);derivative_error=0.;maximum_imag=0.;maximum_speed=0.
    for _ in range(64):
        vector=rng.normal(size=3);vector/=np.linalg.norm(vector);f=rng.uniform(.02,.9)
        state=np.r_[rng.uniform(.5,2.),vector*f];state[1:]*=state[0]
        def flux(s):return np.r_[s[1],pressure(s)[0,:]]
        fd=np.column_stack([(flux(state+np.eye(4)[i]*1e-5)-flux(state-np.eye(4)[i]*1e-5))/(2e-5) for i in range(4)])
        A=flux_jacobian(state);derivative_error=max(derivative_error,np.max(abs(A-fd)))
        ev=np.linalg.eigvals(A);maximum_imag=max(maximum_imag,np.max(abs(ev.imag)));maximum_speed=max(maximum_speed,np.max(abs(ev.real)))
    state=np.array([1.,0,0,0]);A=flux_jacobian(state)
    fd=np.column_stack([(np.r_[(state+np.eye(4)[i]*1e-5)[1],pressure(state+np.eye(4)[i]*1e-5)[0,:]]-np.r_[(state-np.eye(4)[i]*1e-5)[1],pressure(state-np.eye(4)[i]*1e-5)[0,:]])/(2e-5) for i in range(4)])
    zero_error=float(np.max(abs(A-fd)))
    examples=[]
    for directions,weights,mods in [([[1,0,0],[-1,0,0]],[.5,.5],[.7,.7]),([[1,0,0],[-1,0,0]],[.65,.35],[.7,.7]),([[.8,.6,0],[-.8,.6,0]],[.5,.5],[.7,.7]),([[1,0,0],[0,1,0],[-.6,0,.8]],[.35,.3,.35],[.8,.5,.6])]:
      for k,t,x in [(1,2.8,.2),(2,1.4,.4),(3,.9,.1)]:
        exact,base=predict(directions,weights,mods,k,t,x)
        examples.append(dict(directions=directions,weights=weights,modulations=mods,wavenumber=k,time=t,detector=x,exact=exact,shortcut=base,absolute_gap=abs(exact-base)))
    assert derivative_error<1e-8 and zero_error<1e-9 and maximum_imag<1e-10 and maximum_speed<=1
    report=dict(status='prototype_only_unevaluated',analytic_jacobian_max_fd_error=float(derivative_error),zero_flux_jacobian_error=zero_error,maximum_eigenvalue_imaginary_part=float(maximum_imag),maximum_wave_speed=float(maximum_speed),examples=examples)
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
