from math import factorial

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma,gammainc,ndtr
from scipy.stats import poisson

PARAMETER="gain"
TRUE_PARAMETER=1.019
CALIBRATION_SEED=28213923
NOISE_FRACTION=.0025


def ends(b):
    return (-np.inf if b[0] is None else b[0], np.inf if b[1] is None else b[1])


def integral(c,outer,inner):
    l,h=ends(inner)
    l,h=max(l,outer[0]),min(h,outer[1])
    if h<=l:
        return 0.
    return (np.exp(c*h)-np.exp(c*l))/c


def parameters(e,g):
    c=(2*e["ratio"])**.5/g
    sigma=g*np.sqrt(e["variances"])
    a,b=sigma
    root=1/(a*np.sqrt(2*np.pi))
    site=1/(b*np.sqrt(2*np.pi))
    sd=a*b/np.linalg.norm(sigma)
    tilt=e["field"]*c*sd
    space=integral(c,e["shell"],[None,None])
    total=np.exp(tilt**2/2)*ndtr(tilt)*space/(np.sqrt(2*np.pi)*np.linalg.norm(sigma))
    return c,root,site,sd,tilt,space,total


def read(e,g):
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*sum(e["variances"])*g*g
    c,L1,L2,sd,h,part,a=parameters(e,g)
    accepted=-np.expm1(-a)
    if e["observable"]=="shell_available":
        return float(accepted)
    inward=e["sibling_bands"]
    outside=e["other_bands"]
    n=len(inward)
    m=len(outside)
    r=n+m
    P=L2*part
    z0,z1=ends(e["parent_window"])
    u0,u1=P*np.exp(-c*z1),P*np.exp(-c*z0)
    def f(x):
        w=-np.log1p(-x)
        k=gammainc(n-1,u1*x)-gammainc(n-1,u0*x)
        return k*(1-x)**(r-1)*np.exp(-a*x)*(w/x)**(n-1)*w**m
    q=quad(f,0,1,epsabs=1e-11,epsrel=3e-9,limit=250,points=[.05,.21,.57,.85,.98])[0]
    q*=gamma(n-1)/gamma(r)*L1*P*a**m/c
    for I in inward:
        q*=integral(c,e["shell"],I)/part
    for I,B in outside:
        l,h0=ends(B)
        l=max(l,0.)
        t=max(0.,ndtr(h-l/sd)-ndtr(h-h0/sd))
        q*=integral(c,e["shell"],I)/part*t/ndtr(h)
    if e["conditioning"]=="nonempty":
        q/=accepted
    return float(q)


def predict(experiments,g):
    return np.array([read(e,g) for e in experiments])


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.46+.016*i,.72+.021*(i%8)],ratio=.62,
                 inverse_temperature=.081+.01*i) for i in range(31)]


def setup(i):
    return dict(observable="sibling_spectrum",variances=[.51+.123*i,.94-.112*i],ratio=.49+.104*i,
                field=-.91+.43*i,shell=[-.79+.118*i,.74+.269*i],parent_window=[None,None],
                sibling_bands=[[None,None]]*2,other_bands=[],conditioning="all_samples")


def hidden_inputs():
    G={"discrete_shell_population":[],"same_shell_with_background_draws":[],
       "partial_parent_and_energy_bands":[],"nonempty_conditioning_and_calibration":[]}
    for i in range(5):
        e=setup(i)
        G["discrete_shell_population"] += [
            e,dict(e,sibling_bands=[[None,None]]*3),dict(e,sibling_bands=[[None,None]]*4)]
        G["same_shell_with_background_draws"] += [
            dict(e,other_bands=[[[None,None],[None,.67]]]),
            dict(e,sibling_bands=[[None,None]]*3,other_bands=[[[None,None],[.19,None]],
                                                             [[-.54,1.38],[None,None]]])]
        G["partial_parent_and_energy_bands"] += [
            dict(e,parent_window=[-.62,.83]),
            dict(e,parent_window=[None,.22],sibling_bands=[[.23,None],[None,.53]]),
            dict(e,parent_window=[-.15,None],other_bands=[[[.35,None],[.21,1.27]]])]
        G["nonempty_conditioning_and_calibration"] += [
            dict(e,conditioning="nonempty"),dict(e,conditioning="nonempty",parent_window=[-.4,1.23]),
            {k:v for k,v in dict(e,observable="shell_available").items()
             if k not in ("parent_window","sibling_bands","other_bands","conditioning")},
            calibration_inputs()[5*i]]
    return G


def checks():
    g=TRUE_PARAMETER
    e=dict(setup(1),parent_window=[-.53,.47])
    c,L1,L2,sd,h,part,a=parameters(e,g)
    Q=L2*part
    b=np.arange(80)
    k=b[2:]
    b_prob=poisson.pmf(b,a)
    def f(x):
        mu=Q*np.exp(-c*x)
        terms=np.divide((k*(k-1))[:,None],(k[:,None]+b)**2)
        return np.exp(c*x)*(poisson.pmf(k,mu) @ terms @ b_prob)
    manual=L1*quad(f,*e["parent_window"],epsabs=2e-10)[0]
    truth=read(e,g)
    assert abs(manual-truth)<2e-10,(manual,truth)
    conditioned=read(dict(e,conditioning="nonempty"),g)
    assert abs(conditioned*(-np.expm1(-a))-truth)<1e-10
    return dict(window_tally=manual,laplace_probability=truth,nonempty_echo=conditioned,
                ordinary_shell_mean=a)
