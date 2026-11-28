import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,expit

PARAMETER="gain"
TRUE_PARAMETER=1.031
CALIBRATION_SEED=28213823
NOISE_FRACTION=.0026


def state(e,g):
    c=(2*e["ratio"])**.5/g
    sd=g*np.sqrt(e["variances"])
    tilt=c*np.array(e["fields"])-e["window"]
    minimum=-e["detuning"]/c
    return c,sd,tilt,minimum


def bounds(x):
    return (-np.inf if x[0] is None else x[0], np.inf if x[1] is None else x[1])


def integral(st,xb,yb):
    _,s,h,edge=st
    X=min(bounds(xb)[1],s[0]*(15+h[0]*s[0]))
    x=max(bounds(xb)[0],edge)
    if x>=X:
        return 0.
    l,u=bounds(yb)
    u=min(u,s[1]*(15+h[1]*s[1]))
    l=max(l,-X,s[1]*(-15+h[1]*s[1]))
    if l>=u:
        return 0.
    shift=s[0]**2*h[0]
    mul=np.exp(.5*(s[0]*h[0])**2)/(s[1]*np.sqrt(2*np.pi))
    def f(y):
        n=ndtr((X-shift)/s[0])-ndtr((max(x,-y)-shift)/s[0])
        return n*np.exp(-.5*(y/s[1])**2+h[1]*y)
    return mul*quad(f,l,u,epsabs=1e-10,epsrel=3e-9,points=[-x] if l < -x < u else None)[0]


def value(e,g):
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*sum(e["variances"])*g*g
    st=state(e,g)
    c,sd,h,x=st
    Z=integral(st,[None,None],[None,None])
    if e["observable"]=="thermal_band":
        return integral(st,e["family_band"],e["site_band"])/Z
    n=len(e["inside_bands"])
    m=len(e["outside_bands"])
    base=np.exp(-.5*(x/sd[0])**2)/(sd[0]*np.sqrt(2*np.pi))
    def child(band):
        l,u=bounds(band)
        l=max(l,-x)
        if l>=u:
            return 0.
        shift=sd[1]**2*h[1]
        return np.exp(.5*(sd[1]*h[1])**2+h[0]*x)*(ndtr((shift-l)/sd[1])-ndtr((shift-u)/sd[1]))
    S=child([None,None])
    q=np.exp(np.log(S/Z))
    def integrand(z):
        p=expit(z)
        u=expit(-z)
        return p**(n-1)*u**(m+1)
    result=base*q/c*quad(integrand,-55,55,epsabs=3e-11,epsrel=4e-9,
                         points=[-10,0,10])[0]
    for yb in e["inside_bands"]:
        result*=child(yb)/S
    for xb,yb in e["outside_bands"]:
        result*=integral(st,xb,yb)/Z
    return float(result)


def predict(experiments,g):
    return np.array([value(e,g) for e in experiments])


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.46+.019*i,.58+.011*(i%8)],ratio=.63,
                 detuning=.25,inverse_temperature=.086+.009*i) for i in range(31)]


def setup(i):
    return dict(observable="scaled_group",variances=[.45+.129*i,.96-.113*i],ratio=.44+.091*i,
                detuning=-.65+.295*i,window=-.81+.39*i,fields=[-.41+.19*i,.54-.24*i],
                inside_bands=[[None,None],[None,None]],outside_bands=[])


def hidden_inputs():
    G={"ordinary_two_dimensional_cloud":[],"rare_shared_family_diffuse_sites":[],
       "joint_rare_and_ordinary_draws":[],"detuning_and_calibration":[]}
    for i in range(5):
        e=setup(i)
        o={k:v for k,v in dict(e,observable="thermal_band").items()
             if k not in ("inside_bands","outside_bands")}
        G["ordinary_two_dimensional_cloud"] += [
            dict(o,family_band=[.17,1.15],site_band=[-.62,.29]),
            dict(o,family_band=[.72,None],site_band=[None,-.19]),
            dict(o,family_band=[None,.59],site_band=[None,None])]
        G["rare_shared_family_diffuse_sites"] += [
            e,dict(e,inside_bands=[[.17,1.47],[.46,None],[-.25,.96]]),
            dict(e,inside_bands=[[.26,.97],[None,None],[None,None],[None,None]])]
        G["joint_rare_and_ordinary_draws"] += [
            dict(e,outside_bands=[[[.55,None],[None,-.17]]]),
            dict(e,inside_bands=[[.15,None],[None,None],[.21,.93]],outside_bands=[
                [[.36,1.66],[-.42,.59]],[[None,None],[None,None]]]),
            dict(e,outside_bands=[[[None,None],[None,None]],[[None,None],[None,None]]])]
        G["detuning_and_calibration"] += [
            dict(e,detuning=-e["detuning"],window=-e["window"],fields=e["fields"][::-1]),
            dict(e,inside_bands=[[None,-1.52],[None,None]]),
            calibration_inputs()[5*i]]
    return G


def checks():
    g=TRUE_PARAMETER
    e=dict(setup(2),fields=[0.,0.],window=0.,detuning=0.)
    s=state(e,g)
    c,sigma,*_=s
    W=.25+np.arctan(sigma[0]/sigma[1])/(2*np.pi)
    Z=integral(s,[None,None],[None,None])
    assert abs(Z-W)<2e-10
    expect=.5/(c*Z*np.sqrt(2*np.pi)*sigma[0])
    assert abs(expect-value(e,g))<3e-10
    e=dict(e,inside_bands=[[None,None]]*3,outside_bands=[[[None,None],[None,None]]])
    z=quad(lambda t:(.5*t/(Z+.5*t))**3*Z/(Z+.5*t)/t**2,0,np.inf,epsabs=3e-11)[0]
    direct=z/(c*np.sqrt(2*np.pi)*sigma[0])
    result=value(e,g)
    assert abs(result-direct)<2e-10
    return dict(gaussian_cloud_angle=W,computed_mass=Z,leading_sibling_pair=expect,
                four_draw_reference=result,four_draw_absolute_mass_integral=direct)
