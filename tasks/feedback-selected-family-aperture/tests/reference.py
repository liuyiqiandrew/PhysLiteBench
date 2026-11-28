import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

PARAMETER="gain"
TRUE_PARAMETER=1.046
CALIBRATION_SEED=28314011
NOISE_FRACTION=.0039


def normal(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def evaluate(e,g):
    vf,vl=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c=np.sqrt(2*e["ratio"])/g
    coef=[c*np.array(s["fields"])-s["window"] for s in e["settings"]]
    a=e["aperture"]
    mix=a["mixing"]
    slope=a["slope"]
    bias=a["bias"]
    sl=np.sqrt(vl)

    def ordinary(j,bounds,gate=None):
        f,l=coef[j]
        lo,hi=bounds
        start=max(0.,-hi)
        if gate is not None and mix==0:
            start=max(start,gate)
        candidates=[start,np.inf]
        if np.isfinite(lo) and -lo>start:
            candidates.append(-lo)
        if gate is not None and mix:
            for point in [gate/(1-mix) if mix!=1 else start,gate-mix*lo,gate-mix*hi]:
                if start<point<np.inf:
                    candidates.append(point)
        cuts=sorted(set(candidates))

        def allowed(x):
            y=max(lo,-x)
            if gate is not None and mix:
                y=max(y,(gate-x)/mix)
            if y>=hi:
                return 0.
            clip=ndtr((l*vl-y)/sl)-ndtr((l*vl-hi)/sl)
            return normal(x-f*vf,vf)*clip

        result=sum(quad(allowed,left,right,epsabs=2e-12)[0] for left,right in zip(cuts[:-1],cuts[1:]))
        return float(np.exp(.5*(f*f*vf+l*l*vl))*result)

    def local(j,lo,hi):
        b=coef[j][1]
        return 0. if hi<=lo else quad(lambda y:normal(y-b*vl,vl),lo,hi,epsabs=3e-12)[0]*np.exp(b*b*vl/2)

    B0=ordinary(0,[-np.inf,np.inf])
    L0=local(0,0.,np.inf)
    t0,t1=e["root_window"]
    gate_zero=mix==0 and slope==0 and bias==0
    if gate_zero:
        t0=max(t0,0.)
    y0,y1=e["first_band"]
    y0=max(y0,0.)
    if y1<=y0 or t1<=t0:
        return 0.
    lo,hi=e["sibling_band"]
    outside=e["outside_band"]

    def integrate(yp):
        gate=bias+slope*yp
        if mix==0 and gate>0:
            return 0.
        floor=max(0.,gate/mix) if mix else 0.
        good=local(1,max(floor,lo),hi)
        if good==0:
            return 0.
        ordinary_base=ordinary(1,[-np.inf,np.inf],gate)
        tagged_base=local(1,floor,np.inf)
        marked=ordinary(1,outside,gate) if outside is not None else None

        def draw(t):
            u=np.exp(-c*t)
            final=ordinary_base+u*tagged_base
            p=u*good/(final*(B0+u*L0))
            return p if marked is None else p*marked/final
        return normal(yp,vl)*np.exp(coef[0][1]*yp)*quad(draw,t0,t1,epsabs=2e-12)[0]

    cuts=[]
    if slope:
        for threshold in [0.,lo*mix,hi*mix]:
            y=(threshold-bias)/slope
            if y0<y<y1:
                cuts.append(y)
    value=quad(integrate,y0,y1,points=sorted(set(cuts)),epsabs=8e-11)[0]
    return float(normal(0.,vf)*value)


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",ratio=.49,variances=[.51+.057*(i%12),.5+.052*(i%13)],
                 inverse_temperature=float(x)) for i,x in enumerate(np.linspace(.082,.329,107))]


def case(mix=0.,**kw):
    e=dict(observable="feedback_aperture",ratio=.53,variances=[.72,.91],
           settings=[dict(window=-.61,fields=[.67,-.4]),dict(window=.39,fields=[-.42,.72])],
           aperture=dict(mixing=mix,bias=0.,slope=0.),root_window=[-.92,.71],
           first_band=[-.28,1.24],sibling_band=[.11,1.22],outside_band=None)
    e.update(kw)
    return e


def hidden_inputs():
    rng=np.random.default_rng(893754)
    groups={"microscopic_root_sign":[case(),case(root_window=[-1.12,-.19]),
                                     case(root_window=[.11,.92])],
            "updated_aperture":[case(aperture=dict(mixing=.69,bias=-.48,slope=.83)),
                                case(aperture=dict(mixing=0.,bias=-.14,slope=.83))],
            "aperture_with_outside_mark":[case(outside_band=[-.81,-.13]),
                                        case(aperture=dict(mixing=.51,bias=.34,slope=-.47),
                                             outside_band=[-.87,.31])],
            "warm_calibration":calibration_inputs()[::11]}
    for i in range(10):
        common=dict(ratio=float(rng.uniform(.34,.72)),variances=rng.uniform(.52,1.22,2).tolist(),
                    settings=[dict(window=float(t),fields=f.tolist()) for t,f in zip(
                        rng.uniform(-1.04,1.05,2),rng.uniform(-1.03,1.04,(2,2)))],
                    root_window=[-1.34+.071*i,.49+.06*i],
                    first_band=[-.55+.079*i,.87+.062*i])
        groups["microscopic_root_sign"].append(case(**common))
        a=dict(mixing=0. if i%3==0 else float(rng.uniform(.39,1.28)),
               bias=float(rng.uniform(-.86,.78)),slope=float(rng.uniform(-.93,.95)))
        groups["updated_aperture"].append(case(**common,aperture=a))
        groups["aperture_with_outside_mark"].append(case(**common,aperture=a,
                                                         outside_band=[-.94,-.12+.083*i]))
    return groups


def checks():
    g=TRUE_PARAMETER
    absent=evaluate(case(root_window=[-.9,-.21]),g)
    open_before=evaluate(case(root_window=[-.9,-.21],aperture=dict(mixing=0.,bias=-.01,slope=0.)),g)
    full=evaluate(case(),g)
    part=evaluate(case(root_window=[0.,.71]),g)
    assert absent==0 and open_before>0 and abs(full-part)<1e-13
    return dict(excluded=absent,negative_aperture=open_before,positive_cut=[full,part])
