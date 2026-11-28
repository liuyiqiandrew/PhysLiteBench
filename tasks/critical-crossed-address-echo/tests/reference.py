from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

PARAMETER="gain"
TRUE_PARAMETER=1.03
CALIBRATION_SEED=28213871
NOISE_FRACTION=.0025
FU=(None,None)


def bounds(z):
    return (-np.inf if z[0] is None else z[0],np.inf if z[1] is None else z[1])


def density(x,mu,s):
    return np.exp(-.5*((x-mu)/s)**2)/(np.sqrt(2*np.pi)*s)


@lru_cache(maxsize=200)
def Z(mu,s,a,b,C):
    xl,xh=bounds(C[0])
    yl,yh=bounds(C[1])
    zl,zh=bounds(C[2])
    xmin=max(xl,a,mu[0]-13*s[0])
    xmax=min(xh,mu[0]+13*s[0])
    ymin=max(yl,b,mu[1]-13*s[1])
    ymax=min(yh,mu[1]+13*s[1])
    if xmin>=xmax or ymin>=ymax:
        return 0.
    ymin=max(ymin,-xmax-zh)
    if ymin>=ymax:
        return 0.
    def g(y):
        left=max(xmin,-y-zh)
        def f(x):
            lo=max(zl,-x-y)
            p=ndtr((mu[2]-lo)/s[2])-ndtr((mu[2]-zh)/s[2])
            return density(x,mu[0],s[0])*max(0.,p)
        turn=-y-zl
        return density(y,mu[1],s[1])*quad(f,left,xmax,epsabs=1e-10,epsrel=4e-9,
                    points=[turn] if left<turn<xmax else None)[0]
    marks=sorted(set(-x-z for x in (xmin,xmax) for z in (zl,zh) if ymin<-x-z<ymax))
    return quad(g,ymin,ymax,points=marks,epsabs=2e-10,epsrel=4e-9)[0]


def params(e,g):
    chi=np.sqrt(2*e["ratio"])/g
    sv=tuple(g*np.sqrt(e["variances"]))
    t=chi*np.asarray(e["fields"])-e["window"]
    means=tuple(t*np.asarray(sv)**2)
    offset=-np.asarray(e["detunings"])/chi
    return chi,sv,t,means,offset


def value(e,g,direct=False):
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*sum(e["variances"])*g*g
    c,s,h,mu,(a,b)=params(e,g)
    total=Z(mu,s,a,b,(FU,FU,FU))
    if e["observable"]=="thermal_gate":
        return Z(mu,s,a,b,tuple(map(tuple,e["bands"])))/total
    def step(L,H):
        if L>=H:
            return 0.
        return ndtr((mu[2]-L)/s[2])-ndtr((mu[2]-H)/s[2])
    cold=step(-a-b,np.inf)
    n=len(e["internal_bands"])
    ratio=np.prod([step(max(bounds(q)[0],-a-b),bounds(q)[1])/cold for q in e["internal_bands"]])
    row=quad(lambda y:density(y,mu[1],s[1])*step(-a-y,np.inf),b,np.inf,epsabs=8e-11)[0]
    col=quad(lambda x:density(x,mu[0],s[0])*step(-b-x,np.inf),a,np.inf,epsabs=8e-11)[0]
    i,j=np.exp(h[:2]*[a,b]-.5*(h[:2]*s[:2])**2)
    rl,rh=e["row_window"]
    cl,ch=e["column_window"]
    u=i*np.exp(-c*np.array([rh,rl]))
    v=j*np.exp(-c*np.array([ch,cl]))
    def outer(x):
        A,B=total+x*row,col+x*cold
        if direct:
            return quad(lambda y:y**(n-2)/(A+B*y)**n,*v,epsabs=3e-11)[0]*x**(n-2)
        q=B*v/(A+B*v)
        return x**(n-2)*(q[1]**(n-1)-q[0]**(n-1))/(A*B**(n-1)*(n-1))
    answer=quad(outer,*u,epsabs=2e-10,epsrel=5e-9)[0]*(i*j/c**2)*cold**n
    return float(answer*ratio*density(a,0,s[0])*density(b,0,s[1]))


def predict(experiments,g):
    return np.array([value(e,g) for e in experiments])


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.46+.015*i,.66+.021*(i%8),.57],
                 ratio=.59,detunings=[-.15,.29],inverse_temperature=.086+.011*i) for i in range(31)]


def setup(i):
    return dict(observable="crossed_echo",variances=[.52+.134*i,1.04-.111*i,.48+.142*i],ratio=.46+.092*i,
                detunings=[-.71+.29*i,.62-.24*i],window=-.82+.35*i,fields=[-.45+.17*i,.52-.19*i,.13+.17*i],
                row_window=[-.94,1.3],column_window=[-1.12,1.1],internal_bands=[[None,None]]*2)


def hidden_inputs():
    G={"two_competing_rare_strips":[],"joint_gaussian_feasibility":[],"different_internal_states":[],
       "asymmetric_window_and_calibration":[]}
    for i in range(5):
        e=setup(i)
        o={k:v for k,v in dict(e,observable="thermal_gate").items()
             if k not in ("row_window","column_window","internal_bands")}
        G["two_competing_rare_strips"] += [e,dict(e,row_window=[-1.68,.45]),
                                              dict(e,column_window=[.46,1.77])]
        G["joint_gaussian_feasibility"] += [
            dict(o,bands=[[.92,None],[.27,1.32],[None,-.31]]),
            dict(o,bands=[[None,.81],[None,None],[-.19,.58]]),
            dict(o,bands=[[-.14,.96],[.11,1.44],[.24,None]])]
        G["different_internal_states"] += [
            dict(e,internal_bands=[[.33,1.43],[None,None],[.17,.91]]),
            dict(e,internal_bands=[[.32,None],[.71,1.94],[None,None],[.24,1.47]]),
            dict(e,internal_bands=[[None,None]]*3)]
        G["asymmetric_window_and_calibration"] += [
            dict(e,row_window=[1.04,2.1],column_window=[-2.1,-.42]),
            dict(e,column_window=[-.18,.38],fields=[.83,-.62,.73]),calibration_inputs()[5*i]]
    return G


def checks():
    g=TRUE_PARAMETER
    e=dict(setup(2),fields=[0.,0.,0.],detunings=[0.,0.],window=0.)
    c,s,h,mu,(a,b)=params(e,g)
    sd=np.linalg.norm(s)
    expected=1/8+(np.arcsin(s[0]/sd)+np.arcsin(s[1]/sd))/(4*np.pi)
    actual=Z(mu,s,a,b,(FU,FU,FU))
    assert abs(actual-expected)<2e-10
    A=value(e,g)
    B=value(e,g,direct=True)
    assert abs(A-B)<3e-10
    reverse=dict(e,variances=[e["variances"][1],e["variances"][0],e["variances"][2]],
                 row_window=e["column_window"],column_window=e["row_window"])
    assert abs(A-value(reverse,g))<2e-10
    return dict(three_dimensional_orthant=expected,integrated_cloud=actual,
                crossed_echo_analytic=A,crossed_echo_nested=B)
