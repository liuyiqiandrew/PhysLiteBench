import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import binom

PARAMETER="gain"
TRUE_PARAMETER=1.039
CALIBRATION_SEED=28314068
NOISE_FRACTION=.0037
H=np.log(2.)


def pdf(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def evaluate(e,g):
    k,s=e["shape"],e["scale"]
    if e["observable"]=="warm_energy":
        beta=e["inverse_temperature"]
        return g/(1+np.exp(beta*g))+k*g*s/(1+beta*g*s)
    c=H/g
    a=2*g*g/9
    b=k*(g*s/(1+H*s))**2
    t=a+b
    cond=np.sqrt(a*b/t)
    bonus=e["additional"]
    positive=[]
    full=[]
    for j,setting in enumerate(e["settings"]):
        u,v=c*np.array(setting["fields"])-setting["window"]
        m=a*u
        n=b*v
        def pool(lo,hi):
            def routine(z):
                xlow=max(0.,z-hi)
                xhigh=z-lo
                if xhigh<=xlow:
                    return 0.
                center=m+a/t*(z-m-n)
                allowed=ndtr((xhigh-center)/cond)-ndtr((xlow-center)/cond)
                return pdf(z-m-n,t)*allowed
            cuts=[0.,*sorted({x for x in (lo,hi) if 0<x<np.inf}),np.inf]
            return np.exp(.5*(u*m+v*n))*sum(quad(routine,x,y,epsabs=2e-12)[0]
                                              for x,y in zip(cuts[:-1],cuts[1:]))
        def mark(lo,hi):
            lo=max(0.,lo)
            if hi<=lo:
                return 0.
            return np.exp(.5*v*n)*quad(lambda x:pdf(x-n,b),lo,hi,epsabs=2e-12)[0]
        positive.append(mark(0.,np.inf))
        full.append(pool(-np.inf,np.inf))
        if bonus is not None and bonus["setting"]==j:
            outer=mark(*bonus["band"]) if bonus["region"]=="sibling" else pool(*bonus["band"])
    result=0.
    for level in e["levels"]:
        root=g*(level-e["residue"]/3)
        scale=np.exp(-c*root)
        def integrand(d):
            atom=scale*np.exp(-c*d)
            den=full+scale*np.asarray(positive)+atom
            chosen=np.prod(atom/den)*np.exp(c*d)
            if bonus is not None:
                band=outer*scale if bonus["region"]=="sibling" else outer
                chosen*=band/den[bonus["setting"]]
            return chosen
        result+=quad(integrand,*e["line_window"],epsabs=2e-12)[0]/scale
    return float(g*pdf(0,a)*pdf(0,b)*result)


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",shape=.53+.061*(i%12),scale=.49+.077*(i%12),
                 inverse_temperature=float(x)) for i,x in enumerate(np.linspace(.082,.339,109))]


def case(**kw):
    e=dict(observable="line_return",shape=.81,scale=.89,residue=2,levels=[-2,0,1],
           line_window=[-.73,.57],settings=[dict(window=-.61,fields=[.66,-.44]),
                                  dict(window=.4,fields=[-.45,.74])],additional=None)
    e.update(kw)
    return e


def hidden_inputs():
    rng=np.random.default_rng(813718)
    groups={"addressed_lattice_lines":[case(residue=j,levels=[0]) for j in (0,1,2)],
            "selected_liquid_mass":[case(),case(line_window=[-1.1,-.34])],
            "other_family_and_sibling":[case(additional=dict(setting=0,region="outside",band=[-.81,-.12])),
                                     case(additional=dict(setting=1,region="sibling",band=[.18,1.06]))],
            "warm_calibration":calibration_inputs()[::11]}
    for i in range(10):
        e=dict(shape=float(rng.uniform(.54,1.27)),scale=float(rng.uniform(.52,1.41)),
               residue=i%3,settings=[dict(window=float(t),fields=f.tolist()) for t,f in zip(
                   rng.uniform(-1.03,1.05,2),rng.uniform(-1.04,1.06,(2,2)))],
               line_window=[-1.16+.064*i,.39+.071*i])
        groups["addressed_lattice_lines"].append(case(**e,levels=[-i%3-1]))
        groups["selected_liquid_mass"].append(case(**e,levels=[-2,0,1+(i%2)]))
        groups["other_family_and_sibling"].append(case(**e,levels=[-1,0,1],
                           additional=dict(setting=i%2,region="outside" if i%2==0 else "sibling",
                                           band=[-.83,.21+.092*i])))
    return groups


def checks():
    e=case(residue=1,levels=[-2,0,1])
    all_sites=evaluate(e,TRUE_PARAMETER)
    by_line=sum(evaluate(dict(e,levels=[k]),TRUE_PARAMETER) for k in e["levels"])
    assert abs(all_sites-by_line)<2e-13
    n=3000001
    p=binom.pmf(n//3,n,1/3)*np.sqrt(n)
    target=pdf(0,2/9)
    assert abs(p/target-1)<1e-5
    return dict(lattice_populations=[all_sites,by_line],local_binomial_limit=[float(p),float(target)])
