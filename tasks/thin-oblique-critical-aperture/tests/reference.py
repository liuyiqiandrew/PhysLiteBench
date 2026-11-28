import numpy as np
from scipy.integrate import quad

PARAMETER="gain"
TRUE_PARAMETER=1.044
CALIBRATION_SEED=28314186
NOISE_FRACTION=.0034


def pdf(x,v):
    return np.exp(-.5*x*x/v)/np.sqrt(2*np.pi*v)


def evaluate(e,g):
    a,b=g*g*np.array(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(a+b)
    xi=e["center"]
    c=np.sqrt(2*e["ratio"])/g
    A=[]
    lambda_site=[]
    clips=[e["address_band"]]
    for s in e["settings"]:
        m=s["mixing"]
        f,l=np.array(s["fields"])*c-s["window"]
        low,high=s["gate"]
        clips.append([low/(1-m),high/(1-m)])
        if xi==0:
            def slab(x):
                below=max(0.,low-(1-m)*x)
                above=max(0.,high-(1-m)*x)
                return (above-below)/m
            region=quad(slab,0,high/(1-m),points=[low/(1-m)],epsabs=2e-12)[0]
            norm=region*pdf(0.,a)*pdf(0.,b)
        else:
            def line(total):
                x=xi-m*total/(1-m)
                y=total-x
                return pdf(x,a)*pdf(y,b)*np.exp(f*x+l*y)
            norm=(high-low)/(1-m)*quad(line,0,xi*(1-m)/m,epsabs=2e-12)[0]
        A.append(norm)
        lambda_site.append(np.exp((f-l)*xi))
    length=min(x[1] for x in clips)-max(x[0] for x in clips)
    if length<=0:
        return 0.
    def signal(t):
        weights=np.exp(-c*t)*np.array(lambda_site)
        return np.prod(weights/(A+weights))*np.exp(c*t)
    return float(pdf(xi,a)*pdf(-xi,b)*length*quad(signal,*e["line_window"],epsabs=2e-12)[0])


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",ratio=.49,variances=[.51+.054*(i%12),.53+.053*(i%12)],
                 inverse_temperature=float(x)) for i,x in enumerate(np.linspace(.082,.338,105))]


def case(**kw):
    e=dict(observable="thin_echo",ratio=.54,variances=[.76,.83],center=0.,
           settings=[dict(window=-.64,fields=[.66,-.48],gate=[.23,.82],mixing=.48),
                     dict(window=.42,fields=[-.46,.72],gate=[.2,.73],mixing=.63)],
           address_band=[-.25,1.46],line_window=[-.86,.67])
    e.update(kw)
    return e


def hidden_inputs():
    rng=np.random.default_rng(268741)
    groups={"cone_tip_normalization":[case(),case(line_window=[-1.37,-.48])],
            "nonzero_oblique_segments":[case(center=.62),case(center=.03),case(center=.92)],
            "intersection_of_address_bands":[case(address_band=[-1.04,-.18]),case(address_band=[.91,1.12])],
            "warm_calibration":calibration_inputs()[::11]}
    for i in range(12):
        common=dict(ratio=float(rng.uniform(.34,.72)),variances=rng.uniform(.52,1.2,2).tolist(),
                    settings=[dict(window=float(t),fields=f.tolist(),mixing=float(m),
                                   gate=[.15+.03*j,.63+.037*i])
                              for j,(t,f,m) in enumerate(zip(rng.uniform(-1.03,1.04,2),
                                             rng.uniform(-1.04,1.04,(2,2)),rng.uniform(.24,.77,2)))],
                    line_window=[-1.1+.069*i,.31+.073*i])
        groups["cone_tip_normalization"].append(case(**common))
        groups["nonzero_oblique_segments"].append(case(**common,center=.18+.065*i))
        groups["intersection_of_address_bands"].append(case(**common,center=.53 if i%2 else 0.,
                                                              address_band=[.21+.048*i,.93+.059*i]))
    return groups


def checks():
    e=case()
    regular=case(center=.56)
    switch=dict(regular,settings=regular["settings"][::-1])
    a,b=predict([regular,switch],TRUE_PARAMETER)
    expected=evaluate(e,TRUE_PARAMETER)
    assert expected>0 and abs(a-b)<2e-13
    fail=dict(e,settings=[dict(e["settings"][0],mixing=.26,gate=[.14,.26]),
                          dict(e["settings"][1],mixing=.71,gate=[.4,.62])])
    assert evaluate(fail,TRUE_PARAMETER)==0.
    return dict(positive_tip=expected,exchanged_settings=[float(a),float(b)])
