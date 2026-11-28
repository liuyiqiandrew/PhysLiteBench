import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,expit

PARAMETER="gain"
TRUE_PARAMETER=1.027
CALIBRATION_SEED=28214171
NOISE_FRACTION=.0023


def normal(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def answer(e,g):
    v=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*v.sum()
    c=np.sqrt(2*e["ratio"])/g
    tilt=c*np.asarray(e["fields"])-e["window"]
    m=v*tilt

    def density(y):
        return normal(y-m[1],v[1])*ndtr((m[0]-max(0.,-y))/np.sqrt(v[0]))

    B=np.exp(.5*m@tilt)*(quad(density,-np.inf,0.,epsabs=2e-10)[0]
                          +quad(density,0.,np.inf,epsabs=2e-10)[0])

    def mass(j,band):
        low=0. if band[0] is None else max(0.,band[0])
        high=np.inf if band[1] is None else band[1]
        if high<=low:
            return 0.
        t=c*e["tag_fields"][j]-e["window"]
        center=t*v[1]
        return np.exp(-c*e["tag_energies"][j]+.5*t*center)*quad(
            lambda y:normal(y-center,v[1]),low,high,epsabs=1e-11)[0]

    Z=B+sum(mass(j,[None,None]) for j in (0,1))
    j=e["tag"]
    if e["observable"]=="tagged_population":
        return mass(j,e["band"])/Z
    other=e["opposite_band"]

    def f(d):
        log_odds=-c*(e["tag_energies"][j]+d)-np.log(Z)
        weight=np.exp(c*d)*expit(log_odds)**2
        if other is not None:
            weight*=mass(1-j,other)/Z*expit(-log_odds)
        return weight

    return normal(0.,v[1])*quad(f,*e["site_window"],epsabs=3e-11)[0]


def predict(experiments,gain):
    return np.array([answer(e,gain) for e in experiments])


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.45+.021*i,.57+.024*(i%8)],ratio=.58,
                 inverse_temperature=.081+.011*i) for i in range(31)]


def setup(i):
    return dict(observable="tagged_repeat",variances=[.54+.153*i,.94-.111*i],ratio=.48+.121*i,
                window=-.71+.33*i,fields=[.57-.24*i,-.45+.17*i],
                tag_energies=[-.83+.31*i,.68-.287*i],tag_fields=[-.68+.29*i,.57-.24*i],tag=0,
                site_window=[-1.13,1.24],opposite_band=None)


def hidden_inputs():
    G={"rare_label_and_diffuse_tags":[],"shared_temperature_and_opposite":[],"marked_populations":[],
       "windows_and_warm_calibration":[]}
    for i in range(5):
        e=setup(i)
        u={k:v for k,v in dict(e,observable="tagged_population").items() if k not in ["site_window","opposite_band"]}
        G["rare_label_and_diffuse_tags"] += [e,dict(e,tag=1),dict(e,tag_energies=[-.87,-.58])]
        G["shared_temperature_and_opposite"] += [
            dict(e,opposite_band=[None,None]),dict(e,tag=1,opposite_band=[.17,.86]),
            dict(e,site_window=[.18,.77],opposite_band=[.16,None])]
        G["marked_populations"] += [
            dict(u,band=[.12,.91]),dict(u,band=[None,-.13]),dict(u,tag=1,band=[.31,None])]
        G["windows_and_warm_calibration"] += [
            dict(e,site_window=[-.91,-.11],opposite_band=[.24,None]),dict(e,tag=1,site_window=[-.22,.39]),
            calibration_inputs()[4*i]]
    return G


def checks():
    g=TRUE_PARAMETER
    e=setup(2)
    swap=dict(e,tag=1,tag_energies=e["tag_energies"][::-1],tag_fields=e["tag_fields"][::-1])
    v=answer(e,g)
    s=answer(swap,g)
    assert abs(v-s)<2e-10
    assert 0<answer(dict(e,opposite_band=[None,None]),g)<v
    return dict(original=v,swapped=s,opposite_tag=answer(dict(e,opposite_band=[None,None]),g))
