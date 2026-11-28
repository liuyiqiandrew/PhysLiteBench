import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,roots_legendre

PARAMETER = "gain"
TRUE_PARAMETER = 1.042
CALIBRATION_SEED = 28314369
NOISE_FRACTION = .0038
X,W = roots_legendre(31)


def grid(lo,hi):
    return lo+(hi-lo)*(X+1)/2,W*(hi-lo)/2


def pdf(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def evaluate(e,gain):
    v = gain**2*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*sum(v)
    c = np.sqrt(2*e["ratio"])/gain
    parameters = [np.exp(-c*np.asarray(e[k])[::-1]) for k in ["row_window","column_window","site_window"]]
    q,dq = np.array([grid(*band) for band in parameters]).transpose(1,0,2)
    r,s,t = q[0][:,None,None],q[1][None,:,None],q[2][None,None,:]
    state = [1.,r,s,r*s,r*s*t]
    names = ["remote","row","column","corner","tag"]
    use = [dict(setting=0,region="tag",band=None),*e["draws"]]
    factor = np.ones((len(q[0]),len(q[1]),len(q[2])))
    for j,setup in enumerate(e["settings"]):
        h = c*np.asarray(setup["fields"])-setup["window"]
        mu = v*h
        sd = np.sqrt(v)
        co = np.exp(mu*h/2)

        def leaf(x):
            return ndtr((x+mu[2])/sd[2])

        def fraction(region,region_band=None):
            lo,hi = (0.,np.inf) if region_band is None else region_band
            lo = max(0.,lo)
            if hi<=lo:
                return 0.
            if region=="tag":
                return 1.
            if region=="corner":
                return co[2]*(ndtr((hi-mu[2])/sd[2])-ndtr((lo-mu[2])/sd[2]))
            if region=="remote":
                xh,yh = mu[:2]+12*sd[:2]
                xh = min(xh,hi)
                points = [0.,*sorted({p for p in [lo,hi-yh] if 0.<p<xh}),xh]
                total = 0.
                for a,b in zip(points[:-1],points[1:]):
                    x,wx = grid(a,b)
                    l = np.maximum(0.,lo-x)
                    d = np.maximum(0.,np.minimum(hi-x,yh)-l)
                    y = l[:,None]+d[:,None]*(1+X)/2
                    total += (wx*pdf(x-mu[0],v[0])*d/2)@((pdf(y-mu[1],v[1])*leaf(x[:,None]+y))@W)
                return co.prod()*total
            k = 1 if region=="row" else 0
            return co[k]*co[2]*quad(lambda y:pdf(y-mu[k],v[k])*leaf(y),lo,hi,epsabs=3e-12)[0]

        D = sum(w*fraction(name) for w,name in zip(state,names))
        for row in use:
            if row["setting"]==j:
                name = row["region"]
                factor *= state[names.index(name)]*fraction(name,row["band"])/D
    a,b,cw = dq/q**2
    result = a@(factor@cw)@b
    return float(np.prod(pdf(0.,v))*result/c**3)


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",ratio=.47,variances=[.51+.059*(i%13),.52+.057*(i%12),.53+.054*(i%14)],
                 inverse_temperature=float(t)) for i,t in enumerate(np.linspace(.082,.338,106))]


def case(**updates):
    e = dict(observable="site_latch",ratio=.51,variances=[.71,.93,.81],
             row_window=[-.87,.76],column_window=[-.52,.83],site_window=[-.82,.64],
             settings=[dict(fields=[.62,-.38,.42],window=-.53),
                       dict(fields=[-.43,.57,-.31],window=.41)],draws=[])
    e.update(updates)
    return e


def draw(region,setting=1,band=[.13,1.18]):
    return dict(region=region,setting=setting,band=None if region=="tag" else band)


def hidden_inputs():
    groups = {"intersection_site_and_cloud":[case(),case(draws=[draw("tag")]),case(draws=[draw("corner")]),
                                              case(draws=[draw("corner",band=[-.92,-.19])])],
              "crossbar_strips":[case(draws=[draw("row"),draw("column",0)]),
                                 case(draws=[draw("column",0,[-.76,.42])])],
              "different_rows_and_columns":[case(draws=[draw("remote",band=[.39,1.32])]),
                                             case(draws=[draw("remote"),draw("tag",0)])],
              "warm_calibration":calibration_inputs()[::11]}
    rng = np.random.default_rng(740161)
    for i in range(10):
        kw = dict(ratio=float(rng.uniform(.33,.74)),variances=rng.uniform(.5,1.23,3).tolist(),
                  row_window=[-1.37+.077*i,.32+.117*i],column_window=[-1.16+.103*i,.39+.117*i],
                  site_window=[-1.28+.051*i,.44+.096*i],
                  settings=[dict(fields=rng.uniform(-1.04,1.04,3).tolist(),
                                 window=float(rng.uniform(-1.04,1.04))) for _ in range(2)])
        groups["intersection_site_and_cloud"].append(case(**kw,draws=[draw(["corner","tag"][i%2])]))
        groups["crossbar_strips"].append(case(**kw,draws=[draw(["row","column"][i%2],i%2,[-.48,.37+.091*i])]))
        groups["different_rows_and_columns"].append(case(**kw,draws=[draw("remote",i%2,[-.44,.59+.099*i]),
                                                                     draw(["column","row"][i%2],(i+1)%2)]))
    return groups


def checks():
    e = case(draws=[draw("row"),draw("remote",0)])
    def swap(a):
        return [a[1],a[0],a[2]]
    mirrored = dict(e,variances=swap(e["variances"]),row_window=e["column_window"],column_window=e["row_window"],
                    settings=[dict(s,fields=swap(s["fields"])) for s in e["settings"]],
                    draws=[draw("column"),draw("remote",0)])
    a,b = predict([e,mirrored],TRUE_PARAMETER)
    np.testing.assert_allclose(a,b,atol=2e-12)
    impossible = evaluate(case(draws=[draw("corner",band=[-.92,-.12])]),TRUE_PARAMETER)
    assert a>0. and impossible==0.
    return dict(row_column_invariance=[a,b],negative_internal_corner=impossible)
