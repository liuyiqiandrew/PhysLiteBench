import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,roots_legendre

PARAMETER="gain"
TRUE_PARAMETER=1.053
CALIBRATION_SEED=28313911
NOISE_FRACTION=.0033
R,A=roots_legendre(31)


def normal(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def interval(bounds):
    a,b=bounds
    return (a+b)/2+(b-a)*R/2,(b-a)*A/2


def evaluate(e,g):
    v=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*sum(v)
    c=np.sqrt(2*e["ratio"])/g
    k=c*(v[1]+v[2])/v[2]
    B,L=[],[]
    size=sum(v)
    pre=v[0]+v[1]
    inner=pre*v[2]/size
    first=v[0]*v[1]/pre
    remainder=v[1]+v[2]

    for w in e["temperature_windows"]:
        def average(t):
            mu=pre*t/size
            def prefix(p):
                return normal(p-mu,inner)*ndtr(v[0]*p/(pre*np.sqrt(first)))
            region=quad(prefix,0,np.inf,epsabs=2e-11)[0]
            return normal(t+w*size,size)*region
        B.append(np.exp(.5*w*w*size)*quad(average,0,np.inf,epsabs=2e-10)[0])
        def reduced(t):
            return normal(t+w*remainder,remainder)*ndtr(t*np.sqrt(v[1]/(v[2]*remainder)))
        L.append(np.exp(.5*w*w*remainder)*quad(reduced,0,np.inf,epsabs=2e-11)[0])

    t,wt=interval(e["root_window"])
    x,wx=interval(e["line_windows"][0])
    y,wy=interval(e["line_windows"][1])
    W0=np.exp(-c*x)[None,:,None]
    W1=np.exp(-c*y)[None,None,:]
    family=np.exp(-c*t)[:,None,None]
    d0=B[0]+family*L[0]+W0
    d1=B[1]+family*L[1]+W1
    probability=(W0/d0)*(W1/d1)
    j=e["additional"]
    if j is not None:
        i=j["setting"]
        restrict=B[i] if j["region"]=="different_root" else family*L[i]
        probability*=restrict/(d0 if i==0 else d1)
    density=np.exp(k*(x[None,:,None]+y[None,None,:])/2+(c-k)*t[:,None,None])
    norm=normal(0,v[0])*normal(0,v[1])*normal(0,v[2])/2
    return float(norm*wt@((probability*density)@wy@wx))


def predict(experiments,g):
    return np.array([evaluate(e,g) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",ratio=.44,variances=[.47+.063*(i%11),
                 .51+.061*(i%10),.66+.041*(i%12)],inverse_temperature=float(x))
            for i,x in enumerate(np.linspace(.084,.318,109))]


def case(**kw):
    e=dict(observable="site_lock",ratio=.48,variances=[.76,.65,.87],
           temperature_windows=[-.71,.36],root_window=[-.81,.45],
           line_windows=[[-.72,.52],[-.43,.79]],additional=None)
    e.update(kw)
    return e


def hidden_inputs():
    rng=np.random.default_rng(830822)
    groups={"signed_middle_energy":[case(),case(temperature_windows=[.36,-.71],
                     line_windows=[[-.43,.79],[-.72,.52]])],
            "shared_root_diffuse_mass":[case(additional=dict(setting=0,region="same_root_new_middle")),
                                       case(additional=dict(setting=1,region="same_root_new_middle"))],
            "outside_root_mark":[case(additional=dict(setting=0,region="different_root")),
                                 case(additional=dict(setting=1,region="different_root"))],
            "warm_calibration":calibration_inputs()[::11]}
    for i in range(10):
        combo=dict(ratio=float(rng.uniform(.28,.59)),variances=rng.uniform(.51,1.16,3).tolist(),
                   root_window=[-.96+.04*i,.43+.07*i],
                   temperature_windows=rng.uniform(-1.1,1.1,2).tolist(),
                   line_windows=(rng.uniform(-1.1,-.23,2)[:,None]+[0.,1.11]).tolist())
        groups["signed_middle_energy"].append(case(**combo))
        groups["shared_root_diffuse_mass"].append(case(**combo,additional={
                                      "setting":i%2,"region":"same_root_new_middle"}))
        groups["outside_root_mark"].append(case(**combo,additional={
                                      "setting":i%2,"region":"different_root"}))
    return groups


def checks():
    a,b=hidden_inputs()["signed_middle_energy"][:2]
    left,right=predict([a,b],TRUE_PARAMETER)
    assert abs(left-right)<2e-11
    f=evaluate(case(additional=dict(setting=0,region="different_root")),TRUE_PARAMETER)
    s=evaluate(case(additional=dict(setting=0,region="same_root_new_middle")),TRUE_PARAMETER)
    assert f>0 and s>0 and f+s<left
    return dict(swap=[left,right],with_third=[f,s])
