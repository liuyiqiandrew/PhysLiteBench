from math import factorial

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, iv, ive

PARAMETER = "gain"
TRUE_PARAMETER = 1.039
CALIBRATION_SEED = 28314250
NOISE_FRACTION = .0037


def density(x, rates, pi):
    f = np.array([1-x,x])
    r = rates.prod()
    z = np.sqrt(r*f.prod())
    even = r*(pi@f)*(iv(1,2*z)/z if z else 1.)
    odd = (pi@rates)*iv(0,2*z)
    return (even+odd)*np.exp(-rates@f)


def record_density(e, x, rates, pi, symmetric_limit=False):
    n = e["switches"]
    if n==0:
        return 0.
    result = 0.
    dwell = np.array([1-x,x])
    b = e["target_fraction"]
    window = e["first_window"]
    for j in [0,1]:
        if e["terminal"] not in [-1,(j+n)%2]:
            continue
        path = [j] + [(j+i)%2 for i in range(1,n+1)]
        sizes = np.bincount(path,minlength=2)
        leaves = np.bincount(path[:-1],minlength=2)
        probability = pi[j]*np.prod(rates**leaves)*np.exp(-rates@dwell)
        volume = np.prod([dwell[i]**(sizes[i]-1)/factorial(sizes[i]-1) for i in [0,1]])
        gate = 1.
        if window is not None:
            planned = b if j else 1-b
            if planned==0:
                continue
            lo,hi = window
            if sizes[j]==1:
                value = dwell[j]/planned
                gate = float(lo<value<hi)
                if symmetric_limit and value in [lo,hi]:
                    gate = .5
            elif dwell[j]>0:
                a,c = np.clip(planned*np.asarray(window)/dwell[j],0,1)
                gate = betainc(1,sizes[j]-1,c)-betainc(1,sizes[j]-1,a)
        result += probability*volume*gate
    return result


def evaluate(e,gain):
    if e["observable"]=="wire_mean":
        return gain*(e["rate"]*e["time"])
    b = e["target_fraction"]
    if b in [0.,1.]:
        return float(e["switches"]==0 and e["first_window"] is None and e["terminal"] in [-1,int(b)])
    pi = np.array([e["initial_a"],1-e["initial_a"]])
    rates = gain*np.asarray(e["switch_rates"])
    return float(record_density(e,b,rates,pi,True)/density(b,rates,pi))


def finite(e,gain,n):
    b = e["target_fraction"]
    pi = np.array([e["initial_a"],1-e["initial_a"]])
    slow = gain*np.asarray(e["switch_rates"])
    base = np.asarray(e["base_rates"])
    diff = np.asarray(e["increments"])
    z = diff[1]/diff[0]
    factors = n*gain*np.array([z,1/z])
    i = int(np.floor((base+b*diff)@(factors*[1,-1])))

    def skellam(x):
        a,d = (base+x*diff)*factors
        w = 2*np.sqrt(a*d)
        scaled = ive(abs(i),w)
        return np.exp(np.log(scaled)+w-a-d+i*np.log(a/d)/2) if scaled>0 else 0.

    endpoint = pi*np.exp(-slow)*np.array([skellam(0.),skellam(1.)])
    total = endpoint.sum()+quad(lambda x: skellam(x)*density(x,slow,pi),0.,1.,epsabs=1e-12)[0]
    if e["switches"]==0:
        k = e["terminal"]
        return (endpoint.sum() if k<0 else endpoint[k])/total if e["first_window"] is None else 0.
    window = e["first_window"]
    cuts = [] if window is None else sorted({t for x in window for t in [b*x,1-(1-b)*x] if 0<t<1})
    signal = quad(lambda x: skellam(x)*record_density(e,x,slow,pi),0.,1.,points=cuts,epsabs=1e-12)[0]
    return signal/total


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="wire_mean",rate=.31+.067*(i%14),time=float(t))
            for i,t in enumerate(np.linspace(.153,1.385,103))]


def case(**updates):
    e = dict(observable="dwell_record",initial_a=.61,base_rates=[.43,1.27],increments=[.34,1.12],
             switch_rates=[.76,1.38],target_fraction=.49,switches=1,terminal=-1,first_window=[.51,1.])
    e.update(updates)
    return e


def hidden_inputs():
    rng = np.random.default_rng(525708)
    groups = {"microscopic_time_at_endpoint":[case(),case(first_window=[1.,1.37]),case(first_window=[.84,1.34]),
                                                case(first_window=[.48,.98])],
              "endpoint_occupation_atoms":[case(target_fraction=0.,switches=0,first_window=None),
                                           case(target_fraction=1.,switches=0,terminal=0,first_window=None),
                                           case(target_fraction=.49,switches=0,first_window=None)],
              "conditioned_switch_records":[case(switches=2,terminal=1,first_window=[.15,.71]),
                                             case(switches=3,terminal=0,first_window=None)],
              "wire_calibration":calibration_inputs()[::12]}
    for i in range(10):
        kw = dict(initial_a=float(rng.uniform(.14,.86)),base_rates=rng.uniform(.43,1.28,2).tolist(),
                  increments=[float(rng.uniform(.34,.62)),float(rng.uniform(.75,1.13))],
                  switch_rates=rng.uniform(.22,2.42,2).tolist(),target_fraction=float(rng.uniform(.18,.82)),
                  terminal=(i%3)-1)
        groups["microscopic_time_at_endpoint"].append(case(**kw,first_window=[[.45,1.],[1.,1.34]][i%2]))
        groups["conditioned_switch_records"].append(case(**kw,switches=2+i%3,first_window=[.02+.034*i,.56+.08*i]))
        groups["endpoint_occupation_atoms"].append(case(**dict(kw,target_fraction=float(i%2)),
                                                       switches=i%3,first_window=None))
    return groups


def checks():
    g = TRUE_PARAMETER
    e = case()
    midpoint = evaluate(e,g)
    assert abs(midpoint*2-evaluate(dict(e,first_window=None),g))<1e-13
    assert abs(midpoint-evaluate(dict(e,first_window=[1.,1.44]),g))<1e-13
    samples = [finite(e,g,n) for n in [85,700]]
    assert abs(samples[1]-midpoint)<abs(samples[0]-midpoint) and abs(samples[1]-midpoint)<.02
    atom = case(target_fraction=0.,switches=0,first_window=None)
    atoms = [finite(atom,g,n) for n in [85,700]]
    assert 0.<atoms[0]<atoms[1]<1.
    return dict(counter_conditioning_at_time_edge=samples+[midpoint],occupation_atom=atoms+[1.])
