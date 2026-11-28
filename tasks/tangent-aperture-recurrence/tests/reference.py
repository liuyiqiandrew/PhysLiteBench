import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

PARAMETER="gain"
TRUE_PARAMETER=1.041
CALIBRATION_SEED=28314122
NOISE_FRACTION=.0037


def normal(x,v):
    return np.exp(-.5*x*x/v)/np.sqrt(2*np.pi*v)


def evaluate(e,g):
    a,b=g*g*np.array(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(a+b)
    c=np.sqrt(2*e["ratio"])/g
    x0=e["center"]
    rootlo,roothi=e["address_band"]
    if x0==0.:
        rootlo=max(0.,rootlo)
    linelo,linehi=e["line_window"]
    if rootlo>=roothi or linehi<=0.:
        return 0.
    ratios=[]
    tilts=[]
    curv=[]
    centers=[]
    outside=e["outside"]
    for j,setting in enumerate(e["settings"]):
        k=setting["curvature"]
        curv.append(k)
        centers.append(setting["shift"])
        u,v=c*np.array(setting["fields"])-setting["window"]
        p,q=u*a,v*b
        tilts.append(np.exp((u-v)*x0))
        def bulk(lo,hi):
            y0=max(lo,-x0-1/(4*k))
            if hi<=y0:
                return 0.
            def f(y):
                delta=np.sqrt(max(0.,1+4*k*(x0+y)))
                xlo=max(0.,x0+(1-delta)/(2*k))
                xhi=x0+(1+delta)/(2*k)
                return normal(y-q,b)*(ndtr((xhi-p)/np.sqrt(a))-ndtr((xlo-p)/np.sqrt(a)))
            cuts=[y0,hi]
            if y0<k*x0*x0<hi:
                cuts.insert(1,k*x0*x0)
            part=sum(quad(f,l,r,epsabs=2e-12)[0] for l,r in zip(cuts[:-1],cuts[1:]))
            return part*np.exp(.5*(u*p+v*q))
        ratios.append(bulk(-np.inf,np.inf))
        if outside is not None and outside["setting"]==j:
            marked=bulk(*outside["band"])
    baselines=np.array(ratios)

    def signal(t):
        w=np.exp(-c*t)*np.array(tilts)
        den=baselines+w
        raw=np.prod(w/den)*np.exp(c*t)
        return raw if outside is None else raw*marked/den[outside["setting"]]

    def num(u):
        t0=max(linelo,*(k*(u-d)**2 for k,d in zip(curv,centers)))
        return 0. if t0>=linehi else quad(signal,t0,linehi,epsabs=2e-12)[0]

    breaks=[]
    for x in [linelo,linehi]:
        if x>=0.:
            for k,d in zip(curv,centers):
                r=np.sqrt(x/k)
                breaks.extend([d-r,d+r])
    f,l=np.sqrt(curv)
    for s in (-1,1):
        den=f-s*l
        if den:
            breaks.append((f*centers[0]-s*l*centers[1])/den)
    ans=quad(num,rootlo,roothi,points=sorted({u for u in breaks if rootlo<u<roothi}),
             epsabs=2e-12)[0]
    return float(normal(x0,a)*normal(x0,b)*ans)


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",ratio=.49,variances=[.51+.057*(i%12),.52+.051*(i%12)],
                 inverse_temperature=float(x)) for i,x in enumerate(np.linspace(.082,.339,108))]


def case(**kw):
    e=dict(observable="tangent_echo",ratio=.55,variances=[.72,.94],center=0.,
           settings=[dict(curvature=.58,shift=-.41,window=-.62,fields=[.64,-.47]),
                     dict(curvature=.91,shift=.23,window=.42,fields=[-.45,.72])],
           address_band=[-.91,.84],line_window=[-.3,1.22],outside=None)
    e.update(kw)
    return e


def hidden_inputs():
    rng=np.random.default_rng(497328)
    groups={"root_tangent_transition":[case(),case(address_band=[-.79,-.13]),
                                     case(center=.04,address_band=[-.79,-.13])],
            "intersected_moving_apertures":[case(center=.68),case(line_window=[-.83,-.12])],
            "outside_ordinary_area":[case(center=.55,outside=dict(setting=0,band=[-.86,-.12])),
                                    case(outside=dict(setting=1,band=[-.62,.14]))],
            "warm_calibration":calibration_inputs()[::11]}
    for i in range(10):
        common=dict(ratio=float(rng.uniform(.34,.73)),variances=rng.uniform(.53,1.21,2).tolist(),
                    settings=[dict(curvature=float(k),shift=float(d),window=float(t),fields=f.tolist())
                              for k,d,t,f in zip(rng.uniform(.39,1.48,2),rng.uniform(-1.07,1.07,2),
                              rng.uniform(-1.06,1.06,2),rng.uniform(-1.07,1.07,(2,2)))],
                              line_window=[-.58+.074*i,1.1+.045*i])
        groups["root_tangent_transition"].append(case(**common,address_band=[-1.02,.31+.09*i]))
        groups["intersected_moving_apertures"].append(case(**common,center=.11+.078*i))
        groups["outside_ordinary_area"].append(case(**common,center=.0 if i%2 else .58,
                                                     outside=dict(setting=i%2,band=[-.83,.39+.09*i])))
    return groups


def checks():
    e=case(center=.31)
    x=evaluate(e,TRUE_PARAMETER)
    y=evaluate(dict(e,settings=e["settings"][::-1]),TRUE_PARAMETER)
    a,b=predict([case(address_band=[-.72,-.12]),case(center=.03,address_band=[-.72,-.12])],TRUE_PARAMETER)
    assert abs(x-y)<2e-13 and a==0 and b>0
    return dict(reversed_settings=[x,y],root_edge_jump=[float(a),float(b)])
