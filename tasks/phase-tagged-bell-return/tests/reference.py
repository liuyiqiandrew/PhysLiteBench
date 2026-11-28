import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import ndtr

PARAMETER = "gain"
TRUE_PARAMETER = 1.046
CALIBRATION_SEED = 28314211
NOISE_FRACTION = .0037
SIGMA = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
X, W = leggauss(34)


def gauss(x, variance):
    return np.exp(-x*x/(2*variance))/np.sqrt(2*np.pi*variance)


def grid(a, b):
    return a + (b-a)*(1+X)/2, (b-a)*W/2


def evaluate(e, gain):
    vf, vl = gain**2*np.asarray(e["variances"])
    if e["observable"] == "warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c = np.sqrt(2*e["ratio"])/gain
    p = np.array([e["positive_fraction"],1-e["positive_fraction"]])
    pulse = expm(-1j*gain*(np.asarray(e["pulse"]) @ SIGMA.reshape(3,4)).reshape(2,2))
    outside = e["outside"]
    tilt, mass, spin, trace = [], [], [], []
    for j, s in enumerate(e["settings"]):
        f, l = c*np.asarray(s["fields"])-s["window"]
        tilt.append(f-l)
        matrices = [expm(-c*np.tensordot(np.asarray(s["internal"])+m*np.asarray(s["disorder"]),
                                        SIGMA,axes=1)) for m in [1,-1]]
        lengths = np.array([np.trace(a).real for a in matrices])
        trace.append(lengths)
        spin.append([pulse @ a @ pulse.conj().T for a in matrices] if j else matrices)

        def population(low, high):
            def integrand(x):
                a = max(low,-x)
                if a>=high:
                    return 0.
                return gauss(x,vf)*np.exp(f*x+vl*l*l/2)*(ndtr((high-vl*l)/np.sqrt(vl))-
                                                         ndtr((a-vl*l)/np.sqrt(vl)))
            cuts = [0.,*sorted({-k for k in [low,high] if -25<k<0.}),25.]
            return sum(quad(integrand,a,b,epsabs=2e-12)[0] for a,b in zip(cuts[:-1],cuts[1:]))

        mass.append((p@lengths)*population(-np.inf,np.inf))
        if outside is not None and outside["setting"]==j:
            marker = (p@lengths)*population(*outside["band"])
    lo, hi = e["bridge_band"]
    lo = max(lo,0.)
    if lo>=hi:
        return 0.
    x, wx = grid(lo,hi)
    t, wt = grid(*e["line_window"])
    states = np.array([[0.,1.,-np.exp(1j*(e["phase"]+e["phase_slope"]*s)),0.] for s in x])/np.sqrt(2.)
    weights = np.exp(np.array(tilt)[:,None,None]*x[None,:,None]-c*t[None,None,:])
    answer = 0.
    for k in range(2):
        operator = np.kron(spin[0][k],spin[1][k])
        probability = np.einsum("xi,ij,xj->x",states.conj(),operator,states).real
        norm = np.asarray(mass)[:,None,None]+np.asarray(trace)[:,k,None,None]*weights
        value = probability[:,None]*np.prod(weights/norm,axis=0)
        if outside is not None:
            value *= marker/norm[outside["setting"]]
        answer += p[k] * ((wx*gauss(x,vf)*gauss(-x,vl)) @ (value @ (wt*np.exp(c*t))))
    return float(answer)


def predict(experiments, gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.51+.061*(i%11),.49+.053*(i%14)],
                 ratio=.48,inverse_temperature=float(x)) for i,x in enumerate(np.linspace(.084,.339,108))]


def case(**updates):
    e = dict(observable="bell_return",ratio=.62,variances=[.82,.66],positive_fraction=.35,
             settings=[dict(window=-.47,fields=[.64,-.38],internal=[.93,.42,.06],disorder=[.12,-.25,.36]),
                       dict(window=.24,fields=[-.46,.68],internal=[.89,-.42,.11],disorder=[.08,.34,.2])],
             pulse=[.34,-.19,.52],phase=.23,phase_slope=1.13,
             bridge_band=[-.59,1.33],line_window=[-.98,.65],outside=None)
    e.update(updates)
    return e


def hidden_inputs():
    rng = np.random.default_rng(976054)
    base = case()
    aligned = [dict(s,internal=[1.07,0.,0.],disorder=[0.,0.,0.]) for s in base["settings"]]
    groups = {"coherent_detector":[case(settings=aligned,pulse=[0.,0.,0.],phase=p,phase_slope=0.)
                                    for p in [0.,np.pi]],
              "conditional_mark_and_pulse":[case(),case(pulse=[-.68,.51,-.72],phase=-1.05),
                                            case(positive_fraction=.78)],
              "outside_feasible_band":[case(outside=dict(setting=1,band=[-.91,-.11])),
                                       case(bridge_band=[-.91,-.11])],
              "warm_calibration":calibration_inputs()[::12]}
    for i in range(10):
        inp = dict(ratio=float(rng.uniform(.32,.76)),variances=rng.uniform(.5,1.24,2).tolist(),
                   positive_fraction=float(rng.uniform(.2,.8)),phase=float(rng.uniform(-3.15,3.15)),
                   phase_slope=float(rng.uniform(-3.1,3.1)),pulse=rng.uniform(-1.07,1.07,3).tolist(),
                   settings=[dict(window=float(rng.uniform(-1.07,1.07)),fields=rng.uniform(-1.07,1.07,2).tolist(),
                                  internal=rng.uniform(-1.07,1.07,3).tolist(),
                                  disorder=rng.uniform(-1.07,1.07,3).tolist()) for _ in range(2)],
                   line_window=[-1.16+.059*i,.1+.158*i])
        groups["coherent_detector"].append(case(**inp))
        groups["conditional_mark_and_pulse"].append(case(**inp,bridge_band=[.14,.51+.111*i]))
        groups["outside_feasible_band"].append(case(**inp,outside=dict(setting=i%2,band=[-.94,-.42+.121*i])))
    return groups


def checks():
    e = case()
    switched = [dict(s,disorder=[-d for d in s["disorder"]]) for s in e["settings"]]
    rows = [e,dict(e,settings=switched,positive_fraction=1-e["positive_fraction"])]
    a, b = predict(rows,TRUE_PARAMETER)
    assert abs(a-b)<1e-13
    first, second = predict(hidden_inputs()["coherent_detector"][:2],TRUE_PARAMETER)
    c = np.sqrt(2*e["ratio"])/TRUE_PARAMETER
    np.testing.assert_allclose(second/first,np.cosh(2*c*1.07),rtol=2e-13)
    return dict(sign_relabeling=[float(a),float(b)],aligned_bell_phases=[float(first),float(second)])
