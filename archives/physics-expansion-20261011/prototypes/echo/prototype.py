import numpy as np
from scipy.special import jv, roots_hermitenorm


def response(e, gamma, memory=True, cutoff=64):
    tau, u = e['delay'], e['after']
    a = lambda t: -np.expm1(-gamma*t)/gamma
    v = lambda t: 2*e['thermal_speed']**2*(gamma*t+np.expm1(-gamma*t))/gamma**2
    k1, k2, m = e['k1'], e['k2'], e['mode']
    n = np.arange(-cutoff, cutoff+1)
    keep = (m+n*k2) % k1 == 0
    n = n[keep]
    ell = -(m+n*k2)//k1
    q = m+n*k2
    cov = e['thermal_speed']**2*a(tau)*a(u) if memory else 0.
    variance = q*q*v(tau)+m*m*v(u)+2*q*m*cov
    terms = jv(n,m*e['a2']*a(u))*jv(ell,e['a1']*(m*a(tau+u)+n*k2*a(tau)))
    terms = terms*np.exp(-variance/2+1j*(n*e['phase2']+ell*e['phase1']))
    value = terms.sum()
    return float(value.real if e['component']=='cos' else value.imag)


def quadrature(e,gamma,nx=128,ng=72):
    # Independent Gaussian quadrature of the Langevin trajectories, no Bessel expansion.
    tau,u=e['delay'],e['after']
    def moments(t):
        decay=np.exp(-gamma*t)
        drift=(1-decay)/gamma
        # Integral of stationary OU velocity covariance over a square.
        var=2*e['thermal_speed']**2/gamma*(t-drift)
        return drift,var
    a1,v1=moments(tau);a2,v2=moments(u);at,_=moments(tau+u)
    cov=e['thermal_speed']**2*a1*a2
    z,w=roots_hermitenorm(ng);w=w/np.sqrt(2*np.pi)
    x=2*np.pi*np.arange(nx)/nx
    first=np.sqrt(v1)*z if v1 else np.zeros(ng)
    conditional_mean=cov/v1*first if v1 else np.zeros(ng)
    conditional_var=v2-cov*cov/v1 if v1 else v2
    out=0.
    for x0 in x:
        kick1=e['a1']*np.sin(e['k1']*x0+e['phase1'])
        x1=x0+first+a1*kick1
        xf=x0+first+conditional_mean+at*kick1+a2*e['a2']*np.sin(e['k2']*x1+e['phase2'])
        read=np.cos(e['mode']*xf) if e['component']=='cos' else np.sin(e['mode']*xf)
        out+=w@read*np.exp(-.5*e['mode']**2*conditional_var)
    return float(out/nx)


if __name__=='__main__':
    e=dict(delay=2.,after=2.,thermal_speed=.8,k1=1,k2=2,mode=1,a1=.6,a2=.7,phase1=.2,phase2=.7,component='cos')
    for t in [1.,2.,3.]:
        e['delay']=e['after']=t
        print(t,response(e,.17),response(e,.17,False),quadrature(e,.17,256,256),quadrature(e,.17,384,384))
