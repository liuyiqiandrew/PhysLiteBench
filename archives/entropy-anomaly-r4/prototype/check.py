"""Unevaluated entropy-family extension: finite-mass cumulant convergence."""
import json
import time
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.sparse import bmat, csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.linalg import eigvals

@lru_cache(None)
def derivative(points):
    x=np.arange(points)*2*np.pi/points
    k=np.fft.fftfreq(points,1/points)
    D=np.fft.ifft(1j*k[:,None]*np.fft.fft(np.eye(points),axis=0),axis=0).real
    return x,csc_matrix(D)

def reduced(base,contrast,wave,gamma,points=1024):
    x=np.arange(points)*2*np.pi/points
    T=base*(1+contrast*np.cos(x)); Tp=-base*contrast*wave*np.sin(x)
    rho=1/T;rho/=np.sum(rho)
    a=Tp**2/(2*gamma*T); mu=float(rho@a)
    rhs=-gamma*(a-mu)/(wave**2*T)
    ks=np.fft.fftfreq(points,1/points); kh=np.fft.fft(rhs)
    uh=np.zeros(points,complex);uh[1:]=-kh[1:]/ks[1:]**2
    u=np.fft.ifft(uh).real;u-=rho@u
    variance=float(2*rho@((a-mu)*u))
    return dict(mean=mu,shortcut_variance=variance,variance=variance+2*mu)

def finite_mass(base,contrast,wave,gamma,mass,points=47,modes=24):
    x,D=derivative(points);D=wave*D
    T=base*(1+contrast*np.cos(x));Tp=-base*contrast*wave*np.sin(x)
    I=eye(points,format='csc')
    blocks=[[None]*modes for _ in range(modes)]
    for n in range(modes):
        blocks[n][n]=-gamma*n*I
        if n:blocks[n][n-1]=-np.sqrt(mass*base*n)*D
        if n+1<modes:blocks[n][n+1]=-np.sqrt(mass*base*(n+1))*D
        if n>=2:blocks[n][n-2]=gamma*np.sqrt(n*(n-1))*diags(T/base-1)
    G=bmat(blocks,format='lil')
    G[0,:]=0;G[0,:points]=2*np.pi/points
    lu=splu(G.tocsc());b=np.zeros(points*modes);b[0]=1
    p=lu.solve(b)
    W=diags([np.sqrt(np.arange(1,modes))]*2,[-1,1],shape=(modes,modes),format='csc')
    H=np.sqrt(base/mass)*(kron(W,diags(3*Tp/(2*T)))-kron(W@W@W,diags(base*Tp/(2*T*T))))
    # Exact finite-mass entropy differs from integral h only by endpoint terms.
    Hp=H@p;mu=float(np.sum(Hp[:points])*2*np.pi/points)
    b=-mass*(Hp-mu*p);b[0]=0
    p1=lu.solve(b)
    var=float(2*np.sum((H@p1)[:points])*2*np.pi/points)
    coeff=p.reshape(modes,points)
    calorimetric=gamma/mass*np.mean(base*(coeff[0]+np.sqrt(2)*coeff[2])/T-coeff[0])*2*np.pi
    return dict(mean=mu,variance=var,calorimetric_mean=float(calorimetric),density_min=float(coeff[0].min()),stationary_norm=float(np.sum(coeff[0])*2*np.pi/points),poisson_norm=float(np.sum(p1[:points])*2*np.pi/points))

def fluctuation_check(points=63):
    x,D=derivative(points);T=1+.6*np.cos(x);a=.36*np.sin(x)**2/(2*T)
    L=(diags(T)@D@D).toarray()
    values={}
    for lam in [-1.,-.8,-.2,0.,.2,-1.2]:
        values[str(lam)]=float(np.max(eigvals(L+np.diag((lam+lam*lam)*a)).real))
    return dict(counting_convention="E exp(lambda*S)",SCGF=values,symmetry_error=max(abs(values["-0.8"]-values["-0.2"]),abs(values["0.2"]-values["-1.2"])),lambda_minus_one_error=abs(values["-1.0"]),lambda_zero_error=abs(values["0.0"]))

def main():
    t=time.monotonic();cases=[]
    for base,c,w,g in [(1.,.4,1,1.1),(1.2,.6,2,.7),(.8,.2,3,1.5)]:
        r=reduced(base,c,w,g);rows=[]
        for eps in [.02,.01,.005,.0025,.00125]:
            m=eps*g*g/(base*w*w)
            rows.append(dict(epsilon=eps,mass=m,**finite_mass(base,c,w,g,m)))
        extra={name:rows[-3][name]/3-2*rows[-2][name]+8*rows[-1][name]/3 for name in ['mean','variance']}
        refine=finite_mass(base,c,w,g,rows[-1]['mass'],points=63,modes=36)
        cases.append(dict(controls=dict(temperature=base,contrast=c,wavenumber=w,friction=g),reduced=r,finite_mass=rows,extrapolated=extra,refined_last_mass=refine,relative_error={name:abs(extra[name]/r[name]-1) for name in extra},shortcut_relative_error=abs(r['shortcut_variance']/r['variance']-1)))
    small=[]
    for c in [.1,.05,.025]:
        r=reduced(1.,c,1,1.)
        small.append(dict(contrast=c,**r,variance_over_twice_mean=r['variance']/(2*r['mean']),shortcut_over_contrast4=r['shortcut_variance']/c**4))
    out=dict(status='prototype_only_no_task_or_agent_evaluation',cases=cases,small_amplitude=small,fluctuation_relation=fluctuation_check(),elapsed_seconds=time.monotonic()-t)
    path=Path(__file__).with_name('assessment.json');path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
