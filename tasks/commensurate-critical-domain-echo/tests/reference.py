import numpy as np
from scipy.integrate import quad
from scipy.special import expit,ndtr
from scipy.stats import binom

PARAMETER="gain"
TRUE_PARAMETER=1.043
CALIBRATION_SEED=28213897
NOISE_FRACTION=.0026
FULL=[None,None]


def physical_window(b):
    return (-np.inf if b[0] is None else b[0], np.inf if b[1] is None else b[1])


def set_scale(e,g):
    p=np.asarray(e["fractions"])/20
    v=p*(1-p)
    h=e["tilt"]/g*np.asarray(e["fields"])-e["window"]
    return np.sqrt(v),h


def integrate(S,H,X,Y):
    xl,xh=physical_window(X)
    yl,yh=physical_window(Y)
    a=max(0.,xl)
    b=min(xh,S[0]*(14+S[0]*H[0]))
    if a>=b:
        return 0.
    L=max(yl,-b,S[1]*(-14+S[1]*H[1]))
    U=min(yh,S[1]*(14+S[1]*H[1]))
    if L>=U:
        return 0.
    mu=S[0]**2*H[0]
    s=S[0]
    L0=np.exp(.5*(S[0]*H[0])**2)/(np.sqrt(2*np.pi)*S[1])
    def f(y):
        f0=ndtr((b-mu)/s)-ndtr((max(a,-y)-mu)/s)
        return np.exp(-.5*(y/S[1])**2+H[1]*y)*f0
    return L0*quad(f,L,U,points=[-a] if L < -a<U else None,
                   epsabs=2e-11,epsrel=2e-9)[0]


def read(e,g):
    if e["observable"]=="domain_occupation":
        R=e["prior"]*np.exp(-g*e["inverse_temperature"])
        return R/(1-e["prior"]+R)
    S,H=set_scale(e,g)
    A=integrate(S,H,FULL,FULL)
    if e["observable"]=="coordinate_band":
        return integrate(S,H,e["family_band"],e["site_band"])/A
    scale=S[1]
    mu=H[1]*scale**2
    def site(band):
        l,h=physical_window(band)
        l=max(0.,l)
        if l>=h:
            return 0.
        return np.exp(.5*(H[1]*scale)**2)*(ndtr((mu-l)/scale)-ndtr((mu-h)/scale))
    B=site(FULL)
    n,m=len(e["sibling_bands"]),len(e["other_bands"])
    tau=e["tilt"]
    r=tau*np.asarray(e["addressed_levels"])
    V=np.exp(-r)*B
    P=np.sum(np.exp(r)*(V/(A+V))**n*(A/(A+V))**m)/(np.sqrt(2*np.pi)*S[0])
    for b in e["sibling_bands"]:
        P*=site(b)/B
    for x,y in e["other_bands"]:
        P*=integrate(S,H,x,y)/A
    return float(P)


def predict(experiments,g):
    return np.array([read(e,g) for e in experiments])


def calibration_inputs():
    return [dict(observable="domain_occupation",prior=.42+.023*(i%12),inverse_temperature=.25+.04*i)
            for i in range(31)]


def setup(i):
    return dict(observable="domain_echo",fractions=[5+i,12-i],tilt=.8+.264*i,window=-.83+.38*i,
                fields=[.61-.223*i,-.39+.183*i],addressed_levels=[-2,0,2],
                sibling_bands=[[None,None]]*2,other_bands=[])


def hidden_inputs():
    G={"discrete_address_and_normalization":[],"conditional_outside_draws":[],
       "ordinary_count_cloud":[],"single_levels_and_calibration":[]}
    for i in range(5):
        e=setup(i)
        o={k:v for k,v in dict(e,observable="coordinate_band").items()
             if k not in ("addressed_levels","sibling_bands","other_bands")}
        G["discrete_address_and_normalization"] += [
            e,dict(e,addressed_levels=[-4,-3,-1,1,3]),dict(e,addressed_levels=[-1,1],sibling_bands=[
                [None,None],[.21,.72],[None,None],[.09,1.24]])]
        G["conditional_outside_draws"] += [
            dict(e,other_bands=[[[.52,1.31],[None,-.12]]]),
            dict(e,addressed_levels=[-3,1,4],sibling_bands=[[.1,None],[None,.6]],
                 other_bands=[[[None,None],[None,None]],[[.23,.77],[-.61,.25]]])]
        G["ordinary_count_cloud"] += [
            dict(o,family_band=[.19,.87],site_band=[-.39,-.09]),
            dict(o,family_band=[.47,None],site_band=[None,.03]),
            dict(o,family_band=[None,.62],site_band=[.36,None])]
        G["single_levels_and_calibration"] += [
            dict(e,addressed_levels=[-1]),dict(e,addressed_levels=[2]),calibration_inputs()[4*i]]
    return G


def checks():
    g=TRUE_PARAMETER
    e=dict(setup(2),window=0.,fields=[0.,0.],addressed_levels=[0])
    s,h=set_scale(e,g)
    A=.25+np.arctan(s[0]/s[1])/(2*np.pi)
    exact=(.5/(A+.5))**2/(np.sqrt(2*np.pi)*s[0])
    assert abs(exact-read(e,g))<2e-10
    assert abs(integrate(s,h,FULL,FULL)-A)<2e-11
    rho=e["fractions"][0]/20
    tau=e["tilt"]
    p=expit(np.log(rho/(1-rho))+tau)
    rate=rho*np.log(rho/p)+(1-rho)*np.log((1-rho)/(1-p))
    d=2
    values=[(N,np.exp(np.log(N)/2+N*rate+binom.logpmf(int(N*rho)+d,N,p)))
            for N in (400,1600,6400)]
    lim=np.exp(tau*d)/(np.sqrt(2*np.pi)*s[0])
    assert abs(values[-1][1]/lim-1)<.006
    return dict(zero_address_echo=exact,finite_lattice_intensity=values,
                limiting_lattice_intensity=lim,cloud_probability=A)
