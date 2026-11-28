import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,roots_legendre

PARAMETER="gain"
TRUE_PARAMETER=1.032
CALIBRATION_SEED=28214197
NOISE_FRACTION=.0024
X,W=roots_legendre(28)


def density(x,var):
    return np.exp(-x*x/(2*var))/np.sqrt(2*np.pi*var)


def response(e,g):
    v=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*v.sum()
    c=np.sqrt(2*e["ratio"])/g
    tag=e["tag"]
    other=1-tag
    outside=e["outside"]

    def tail(coef,bounds):
        lo,hi=bounds
        tilt=coef*v[1]
        if hi<=lo:
            return 0.
        return np.exp(.5*coef*tilt)*quad(lambda y:density(y-tilt,v[1]),lo,hi,epsabs=2e-10)[0]

    def aperture(bounds):
        a,b=bounds
        width=(b-a)/2
        return (a+b)/2+width*X,width*W

    weights=[]
    root=[]
    for bounds in e["root_windows"]:
        d,w=aperture(bounds)
        f=w*np.exp(c*d)
        root.append(np.exp(-c*d))
        weights.append(f/f.sum())
    root=[root[0][:,None,None],root[1][None,:,None]]
    d,w=aperture(e["site_window"])
    S=root[tag]*np.exp(-c*d)
    norm=[]
    for setting in e["settings"]:
        a,b=c*np.asarray(setting["fields"])-setting["window"]
        def f(y):
            return density(y-b*v[1],v[1])*ndtr((a*v[0]-max(0.,-y))/np.sqrt(v[0]))
        B=np.exp(.5*(a*a*v[0]+b*b*v[1]))*(quad(f,-np.inf,0.,epsabs=2e-10)[0]
                                          +quad(f,0.,np.inf,epsabs=2e-10)[0])
        children=c*np.asarray(setting["tag_fields"])-setting["window"]
        L=[tail(t,(0.,np.inf)) for t in children]
        norm.append(B+root[0]*L[0]+root[1]*L[1]+S)
    result=S*S/(norm[0]*norm[1])
    if outside is not None:
        j=outside["setting"]
        setting=e["settings"][j]
        lo,hi=outside["band"]
        D=tail(c*setting["tag_fields"][other]-setting["window"],(max(0.,lo),hi))
        result*=root[other]*D/norm[j]
    return float(weights[0]@(result@(w*np.exp(c*d)))@weights[1]*density(0.,v[1]))


def predict(experiments,g):
    return np.array([response(e,g) for e in experiments])


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.45+.021*i,.58+.023*(i%7)],ratio=.59,
                 inverse_temperature=.08+.011*i) for i in range(33)]


def setup(i):
    return dict(observable="heralded_return",variances=[.52+.15*i,.93-.106*i],ratio=.47+.122*i,
                root_windows=[[-1.97,-.22],[.11,1.94]],
                settings=[dict(window=-.73+.33*i,fields=[.51-.2*i,-.52+.24*i],tag_fields=[-.51,.48]),
                          dict(window=.68-.26*i,fields=[-.54+.21*i,.32],tag_fields=[.67-.16*i,-.35])],
                tag=0,site_window=[-1.16,1.21],outside=None)


def hidden_inputs():
    G={"asymmetric_herald_windows":[],"two_temperature_masses":[],"countertag_aperture":[],
       "narrow_heralds_and_calibration":[]}
    for i in range(5):
        e=setup(i)
        G["asymmetric_herald_windows"] += [
            e,dict(e,tag=1),dict(e,root_windows=[[-1.82,-.56],[-.36,1.76]])]
        G["two_temperature_masses"] += [
            dict(e,settings=[e["settings"][0]]*2),dict(e,settings=e["settings"][::-1]),
            dict(e,tag=1,site_window=[-.73,.12])]
        G["countertag_aperture"] += [
            dict(e,outside=dict(setting=0,band=[.19,1.08])),
            dict(e,outside=dict(setting=1,band=[.27,1.25]),tag=1),
            dict(e,outside=dict(setting=1,band=[-.72,-.11]))]
        G["narrow_heralds_and_calibration"] += [
            dict(e,root_windows=[[-.83,-.43],[.12,.53]],site_window=[.11,.78]),
            dict(e,root_windows=[[-2.02,-1.54],[.83,1.23]],tag=1),calibration_inputs()[4*i]]
    return G


def checks():
    e=setup(1)
    g=TRUE_PARAMETER
    val=response(e,g)
    swapped=response(dict(e,settings=e["settings"][::-1]),g)
    p,q=e["site_window"]
    pieces=sum(response(dict(e,site_window=a),g) for a in [[p,0.],[0.,q]])
    assert abs(val-swapped)<2e-12 and abs(val-pieces)<2e-10
    return dict(original=val,exchanged=swapped,site_window_partition=pieces)
