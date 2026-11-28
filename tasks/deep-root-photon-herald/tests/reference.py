import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, quad_vec
from scipy.special import ndtr

PARAMETER = "gain"
TRUE_PARAMETER = 1.041
CALIBRATION_SEED = 28314233
NOISE_FRACTION = .0036
X, W = leggauss(27)


def nodes(lo, hi):
    return lo+(hi-lo)*(X+1)/2, W*(hi-lo)/2


def gaussian(x,v):
    return np.exp(-.5*x*x/v)/np.sqrt(2*np.pi*v)


def evaluate(e,gain):
    vf,vl = gain**2*np.asarray(e["variances"])
    if e["observable"] == "warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c = np.sqrt(2*e["ratio"])/gain
    d = e["depth"]
    first, second = e["settings"]
    left = max(first["gate"][0],second["gate"][0],e["line_window"][0])
    right = min(first["gate"][1],second["gate"][1],e["line_window"][1])
    if right<=left:
        return 0.
    mass,coupling = [],[]
    for setting in [first,second]:
        a,b = c*np.array(setting["fields"])-setting["window"]
        f = lambda x: gaussian(x,vf)*np.exp(a*x+.5*vl*b*b)*ndtr((x+vl*b)/np.sqrt(vl))
        mass.append(quad(f,0.,25.,epsabs=1e-12)[0])
        coupling.append(np.exp(d*(b-a)))
    root, dr = nodes(*e["root_window"])
    count_prefactor = e["retention"]*gaussian(d,vl)/c
    frequency = count_prefactor*np.exp(-c*root)
    threshold0, threshold1 = np.exp(c*np.array([first["gate"],second["gate"]]))
    retained, dt = nodes(np.exp(c*left),np.exp(c*right))
    one = np.exp(-frequency*(threshold0[1]-threshold0[0]))
    w0,w1 = np.array(coupling)[:,None]/retained
    den = mass[1]+w1
    band = [(threshold1[0],threshold0[0]),(threshold0[1],threshold1[1])]
    bins, widths = [],[]
    for lo,hi in band:
        if lo<hi:
            points,sz = nodes(lo,hi)
            bins.extend(coupling[1]/points)
            widths.extend(sz)
    rates = np.asarray(bins)
    widths = np.asarray(widths)
    terms = dr * gaussian(-d,vf) * np.exp(c*root)*one*frequency
    spectrum = dt*w0*w1/(mass[0]+w0)
    power = 1+e["outside"]

    def transform(u):
        count = -np.expm1(-u*rates/den[:,None]) @ widths
        moment = np.exp(-frequency[:,None]*count) * np.exp(-u)*u**(power-1)/den**power
        return terms @ (moment @ spectrum)

    answer = quad_vec(transform,0.,np.inf,epsabs=1e-12)[0]
    return float(answer*mass[1] if e["outside"] else answer)


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",ratio=.51,variances=[.53+.059*(i%11),.55+.058*(i%12)],
                 inverse_temperature=float(t)) for i,t in enumerate(np.linspace(.083,.338,103))]


def case(**updates):
    e = dict(observable="herald_return",variances=[.79,.68],ratio=.52,depth=.64,retention=.85,
             root_window=[-.81,.93],line_window=[-.84,.59],outside=False,
             settings=[dict(fields=[.62,-.52],window=-.53,gate=[-.39,.61]),
                       dict(fields=[-.48,.6],window=.43,gate=[-1.16,1.38])])
    e.update(updates)
    return e


def hidden_inputs():
    e = case()
    s0,s1 = e["settings"]
    groups = {"shared_visibility_and_count":[e,case(settings=[s0,dict(s1,gate=s0["gate"])]),
                                             case(line_window=[.82,1.12])],
              "extra_poisson_competitors":[case(outside=True),case(settings=[dict(s0,gate=[-.8,.65]),s1])],
              "deep_root_response":[case(depth=.23),case(depth=1.14,root_window=[-1.21,-.54])],
              "warm_calibration":calibration_inputs()[::12]}
    rng = np.random.default_rng(216576)
    for i in range(10):
        setting0 = dict(fields=rng.uniform(-1.,1.,2).tolist(),window=float(rng.uniform(-1.,1.)),
                        gate=[-.83+.075*i,.21+.093*i])
        setting1 = dict(fields=rng.uniform(-1.,1.,2).tolist(),window=float(rng.uniform(-1.,1.)),
                        gate=[-1.41+.15*(i%3),1.43-.089*i])
        common = dict(variances=rng.uniform(.52,1.22,2).tolist(),ratio=float(rng.uniform(.34,.73)),
                      depth=float(rng.uniform(.23,1.15)),retention=float(rng.uniform(.13,1.06)),
                      root_window=[-1.32+.112*i,.39+.094*i],settings=[setting0,setting1])
        groups["shared_visibility_and_count"].append(case(**common,line_window=[-.72,.16+.105*i]))
        groups["extra_poisson_competitors"].append(case(**common,outside=True))
        groups["deep_root_response"].append(case(**dict(common,depth=1.15-.09*i)))
    return groups


def checks():
    e = case(outside=True)
    s0,s1 = e["settings"]
    answer = evaluate(e,TRUE_PARAMETER)
    no_competitors = evaluate(dict(e,settings=[s0,dict(s1,gate=s0["gate"])]),TRUE_PARAMETER)
    assert 0.<answer<.95*no_competitors
    assert answer<evaluate(dict(e,outside=False),TRUE_PARAMETER)
    assert evaluate(dict(e,line_window=[.67,.89]),TRUE_PARAMETER)==0.
    return dict(outside_with_and_without_competitors=[answer,no_competitors])
