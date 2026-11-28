import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, quad_vec
from scipy.special import gamma, ndtr

PARAMETER = "gain"
TRUE_PARAMETER = 1.041
CALIBRATION_SEED = 28314257
NOISE_FRACTION = .0038
R,W = leggauss(29)


def density(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def grid(lo,hi):
    return lo+(hi-lo)*(R+1)/2,(hi-lo)*W/2


def evaluate(e,g):
    vf,vl = g*g*np.array(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c = np.sqrt(2*e["ratio"])/g
    power = 1/e["amplification"]
    h,l = c*np.asarray(e["critical_fields"])-e["critical_window"]
    Lcoef = c*e["cold_field"]-e["cold_window"]
    sl = np.sqrt(vl)
    def outer(x):
        return density(x,vf)*np.exp(h*x+l*l*vl/2)*ndtr((x+l*vl)/sl)
    B = quad(outer,0.,25.,epsabs=2e-12)[0]
    L = quad(lambda y:density(y,vl)*np.exp(Lcoef*y),0.,25.,epsabs=2e-12)[0]
    scale = L*(density(0.,vf)*gamma(1-power)/c)**(1/power)
    z,dz = grid(*e["root_window"])
    t,dt = grid(*e["leaf_window"])
    first = np.exp(-c*z/power)
    second = np.exp(-c*t)
    tag = first[:,None]*second
    a = L*first[:,None]+tag
    norm = a+scale
    q = a/norm
    k = (scale/norm)**power
    outside = e["outside_band"]
    if outside is None:
        band = 1.
    else:
        a0,b0 = np.maximum(outside,0.)
        band = quad(lambda y:density(y,vl)*np.exp(Lcoef*y),a0,b0,epsabs=2e-12)[0]/L
    weights = dz[:,None]*dt*tag/norm

    def transform(s):
        p = np.exp(-q*s-k*s**power)
        if outside is not None:
            p *= 1-q*s
        return np.sum(weights*p)

    total = quad_vec(transform,0.,np.inf,epsabs=2e-12,epsrel=2e-10)[0]
    return float(band*total*density(0.,vf)*density(0.,vl)/B)


def predict(experiments,g):
    return np.array([evaluate(e,g) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.52+.064*(i%11),.53+.052*(i%14)],ratio=.49,
                 inverse_temperature=float(beta)) for i,beta in enumerate(np.linspace(.082,.338,107))]


def case(**updates):
    e = dict(observable="frozen_echo",variances=[.76,.9],ratio=.57,amplification=1.76,
             cold_field=.31,cold_window=-.64,critical_fields=[-.54,.63],critical_window=.27,
             root_window=[-.88,.73],leaf_window=[-1.04,.67],outside_band=None)
    e.update(updates)
    return e


def hidden_inputs():
    rng = np.random.default_rng(715806)
    groups = {"tagged_root_atom":[case(),case(leaf_window=[-1.46,-.41]),case(root_window=[.49,1.28])],
              "non_self_averaging_roots":[case(amplification=1.22),case(amplification=2.41)],
              "cold_outside_cloud":[case(outside_band=[-.78,.82]),case(outside_band=[-.98,-.21])],
              "warm_calibration":calibration_inputs()[::12]}
    for i in range(10):
        c = dict(variances=rng.uniform(.52,1.22,2).tolist(),ratio=float(rng.uniform(.33,.74)),
                 amplification=float(rng.uniform(1.22,2.41)),cold_field=float(rng.uniform(-1.02,1.02)),
                 cold_window=float(rng.uniform(-1.02,1.02)),critical_fields=rng.uniform(-1.02,1.02,2).tolist(),
                 critical_window=float(rng.uniform(-1.02,1.02)),root_window=[-1.44+.069*i,.41+.108*i])
        groups["tagged_root_atom"].append(case(**c,leaf_window=[-1.43+.073*i,-.39+.131*i]))
        groups["non_self_averaging_roots"].append(case(**dict(c,amplification=1.23+.124*i)))
        groups["cold_outside_cloud"].append(case(**c,outside_band=[-.71+.079*i,.43+.093*i]))
    return groups


def checks():
    e = case()
    full = evaluate(e,TRUE_PARAMETER)
    split = sum(evaluate(dict(e,root_window=x),TRUE_PARAMETER) for x in [[-.88,0.],[0.,.73]])
    np.testing.assert_allclose(full,split,atol=2e-12)
    outside = evaluate(dict(e,outside_band=[0.,1.42]),TRUE_PARAMETER)
    assert 0.<outside<full
    assert abs(evaluate(dict(e,outside_band=[-.88,-.35]),TRUE_PARAMETER))<1e-15
    return dict(root_window_partition=[full,split],conditional_outside=outside)
