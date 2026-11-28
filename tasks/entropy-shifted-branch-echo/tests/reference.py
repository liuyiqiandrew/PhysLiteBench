import numpy as np
from scipy.integrate import quad
from scipy.special import expit, ndtr

PARAMETER="gain"
TRUE_PARAMETER=1.029
CALIBRATION_SEED=28214057
NOISE_FRACTION=.0026


def bounds(a):
    return [float("-inf") if a[0] is None else a[0],float("inf") if a[1] is None else a[1]]


def density(x,mean,var):
    return np.exp(-.5*(x-mean)**2/var)/np.sqrt(2*np.pi*var)


def answer(e,g):
    v=g*g*np.array(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*v.sum()
    c=np.sqrt(2*e["ratio"])/g
    d=np.array(e["entropy_shifts"])/c
    a=c*np.array(e["fields"])-e["window"]
    m=v*a
    var=v[0]+v[1]
    sigma=np.sqrt(v[0]*v[1]/var)
    tilt=np.exp(.5*v*a*a)

    def ordinary(s):
        mx=m[0]+(s-m[0]-m[1])*v[0]/var
        return (density(s,m[0]+m[1],var)*ndtr((mx+d[0])/sigma)
                *ndtr((m[2]+sum(d)+s)/np.sqrt(v[2]))*np.prod(tilt))

    B=quad(ordinary,-d[0]-d[1],np.inf,epsabs=2e-10)[0]

    def leaf(band):
        lo,hi=bounds(band)
        lo=max(lo,-d[2])
        return tilt[2]*max(0.,ndtr((hi-m[2])/np.sqrt(v[2]))-ndtr((lo-m[2])/np.sqrt(v[2])))

    L=leaf([None,None])
    K=[leaf(b) for b in e["leaf_bands"]]
    lo,hi=bounds(e["first_band"])
    lo=max(lo,-d[0])
    if lo>=hi or not all(K):
        return 0.
    t=-d[0]-d[1]

    def marked(x):
        y=t-x
        log_scale=a[0]*x+a[1]*y+np.log(L/B)
        def f(u):
            log_odds=log_scale-c*u
            share=expit(log_odds)
            numerator=np.exp(c*u)*(share/L)**2
            return numerator*expit(-log_odds) if e["outside"] else numerator
        z=quad(f,*e["prefix_window"],epsabs=1e-11)[0]
        return density(x,0.,v[0])*density(y,0.,v[1])*np.prod(K)*z

    return quad(marked,lo,hi,epsabs=8e-10,epsrel=3e-9)[0]


def predict(experiments,gain):
    return np.array([answer(e,gain) for e in experiments])


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.45+.022*i,.57+.027*(i%7),.56+.02*(i%6)],
                 ratio=.63,inverse_temperature=.081+.011*i) for i in range(31)]


def setup(i):
    return dict(observable="branch_echo",variances=[.51+.17*i,.94-.112*i,.57+.12*i],
                ratio=.48+.13*i,window=-.79+.37*i,entropy_shifts=[.68-.26*i,-.41+.14*i,.51-.21*i],
                fields=[-.59+.27*i,.46-.20*i,-.27+.17*i],
                first_band=[None,None],prefix_window=[-1.31+.13*i,1.06+.13*i],
                leaf_bands=[[None,None],[None,None]],outside=False)


def hidden_inputs():
    groups={"ordinary_and_rare_clouds":[],"entropy_shifted_first_feasibility":[],
            "marked_leaf_and_outside":[],"cutoff_and_calibration":[]}
    for i in range(5):
        e=setup(i)
        v=e["entropy_shifts"]
        groups["ordinary_and_rare_clouds"] += [e,dict(e,outside=True),
            dict(e,entropy_shifts=[-.81,.57,v[2]],outside=True)]
        groups["entropy_shifted_first_feasibility"] += [
            dict(e,first_band=[-.82,-.08],entropy_shifts=[.89,v[1],v[2]]),
            dict(e,first_band=[-.82,-.08],entropy_shifts=[-.61,v[1],v[2]]),
            dict(e,first_band=[-.31,1.01],entropy_shifts=[-.26,.35,.62])]
        groups["marked_leaf_and_outside"] += [
            dict(e,leaf_bands=[[-.64,.72],[.19,1.63]],outside=True),
            dict(e,leaf_bands=[[.15,None],[None,.53]],outside=False),
            dict(e,leaf_bands=[[-.59,-.1],[.21,1.1]],entropy_shifts=[.27,-.47,.82],outside=True)]
        groups["cutoff_and_calibration"] += [
            dict(e,prefix_window=[.22,.86],first_band=[.13,1.34],outside=True),
            dict(e,prefix_window=[-.53,.25],first_band=[None,.45],outside=True),
            calibration_inputs()[5*i]]
    return groups


def checks():
    g=TRUE_PARAMETER
    e=setup(1)
    step=answer(e,g)
    three=answer(dict(e,outside=True),g)
    assert 0.<three<step
    p,q=e["prefix_window"]
    parts=sum(answer(dict(e,prefix_window=w),g) for w in [[p,0.],[0.,q]])
    assert abs(step-parts)<1e-11
    forbid=dict(e,entropy_shifts=[-.66,0.,0.],first_band=[None,0.])
    assert answer(forbid,g)==0.
    return dict(pair=step,with_outside_draw=three,prefix_window_partition=parts,
                forbidden_first_prefix=answer(forbid,g))
