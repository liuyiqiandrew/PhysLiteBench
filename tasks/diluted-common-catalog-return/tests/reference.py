import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

PARAMETER="gain"
TRUE_PARAMETER=1.043
CALIBRATION_SEED=28313973
NOISE_FRACTION=.0036


def normal(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def evaluate(e,g):
    vf,vl=g*g*np.array(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c=np.sqrt(2*e["ratio"])/g
    private=e["private_shifts"]
    edge=-min(private)/c
    coeff=[np.array(s["fields"])*c-s["window"] for s in e["settings"]]
    B=[]
    for shift,(a,b) in zip(private,coeff):
        mw=-shift/c
        def cell(x):
            return normal(x-vf*a,vf)*ndtr((x+vl*b-mw)/np.sqrt(vl))
        B.append(np.exp(.5*(vf*a*a+vl*b*b))*quad(cell,mw,np.inf,epsabs=2e-12)[0])
    lower,upper=e["bridge_band"]
    lower=max(lower,edge+e["shared_shifts"][1]/c)
    if upper<=lower:
        return 0.
    shifted=np.array([s==min(private) for s in private])
    outside=e["outside_setting"]

    def bridge(x):
        value=np.array([a*x+b*(edge-x) for a,b in coeff])
        def chosen(t):
            w=np.exp(value-c*t)
            den=B+w*shifted
            p=np.prod(w/den)*np.exp(c*t)
            return p if outside==-1 else p*B[outside]/den[outside]
        amt=quad(chosen,*e["line_window"],epsabs=2e-12)[0]
        return float(normal(x,vf)*normal(edge-x,vl)*amt)
    return float(quad(bridge,lower,upper,epsabs=2e-12)[0])


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.52+.052*(j%12),.51+.055*(j%11)],
                 ratio=.44,inverse_temperature=float(x)) for j,x in enumerate(np.linspace(.081,.328,108))]


def case(**kw):
    e=dict(observable="shared_site",ratio=.53,variances=[.79,.84],
           private_shifts=[.42,.9],shared_shifts=[.28,-.62],
           settings=[dict(window=-.61,fields=[.69,-.41]),dict(window=.38,fields=[-.46,.81])],
           bridge_band=[-.87,.8],line_window=[-.73,.57],outside_setting=-1)
    e.update(kw)
    return e


def hidden_inputs():
    rng=np.random.default_rng(274697)
    groups={"shared_leaf_entropic_gate":[case(bridge_band=[-.87,-.5]),case(shared_shifts=[-.47,.61]),
                                        case(shared_shifts=[-.33,.49],bridge_band=[-.81,.06])],
            "equal_catalog_scales":[case(private_shifts=[.42,.42]),case(private_shifts=[.42,.42],
                                                                        outside_setting=0)],
            "one_finite_atom":[case(),case(outside_setting=0),case(outside_setting=1)],
            "warm_calibration":calibration_inputs()[::11]}
    for i in range(10):
        d=float(rng.uniform(-.33,.61))
        leaf=float(rng.uniform(-.57,.54))
        common=dict(ratio=float(rng.uniform(.32,.71)),variances=rng.uniform(.52,1.21,2).tolist(),
                    shared_shifts=[d-leaf-.36,leaf],
                    settings=[dict(window=float(t),fields=f.tolist()) for t,f in zip(
                        rng.uniform(-1.07,1.07,2),rng.uniform(-1.09,1.08,(2,2)))],
                    line_window=[-.96+.045*i,.46+.04*i])
        shifts=[d,d+.34] if i%2==0 else [d+.44,d]
        groups["shared_leaf_entropic_gate"].append(case(**common,private_shifts=shifts,
                                                       bridge_band=[-1.09,.38+.073*i]))
        groups["equal_catalog_scales"].append(case(**common,private_shifts=[d,d],
                                                    outside_setting=(i%3)-1))
        groups["one_finite_atom"].append(case(**common,private_shifts=shifts,outside_setting=i%2))
    return groups


def checks():
    e=case(bridge_band=[-.82,-.51])
    original=evaluate(e,TRUE_PARAMETER)
    shift=evaluate(dict(e,shared_shifts=[-.28,-.06]),TRUE_PARAMETER)
    assert original>0 and shift==0
    bigger=evaluate(case(outside_setting=1),TRUE_PARAMETER)
    base=evaluate(case(),TRUE_PARAMETER)
    assert abs(bigger-base)<1e-13
    return dict(shifted_support=[original,shift],weak_atom_outside=[bigger,base])
