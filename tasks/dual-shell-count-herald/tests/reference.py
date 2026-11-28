from itertools import product
from math import factorial

import numpy as np
from scipy.integrate import quad

PARAMETER="gain"
TRUE_PARAMETER=1.025
CALIBRATION_SEED=28313887
NOISE_FRACTION=.0035
MASK=[(1,0),(0,1),(1,1)]


def evaluate(e,g):
    v0,v1=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(v0+v1)
    c=np.sqrt(2*e["ratio"])/g
    v=v0*v1/(v0+v1)
    norm=np.sqrt(2*np.pi*(v0+v1))*np.sqrt(2*np.pi*v)

    def measure(req,bridge,tag):
        if tag:
            l=max(b[0] for j,b in req)
            h=min(b[1] for j,b in req)
            return max(0.,np.exp(c*h)-np.exp(c*l))/(c*np.sqrt(2*np.pi*v1))
        lo,hi=(0.,12*np.sqrt(v)) if bridge is None else (max(0.,bridge[0]),bridge[1])
        if hi<=lo:
            return 0.
        f=[e["fields"][j] for j,b in req]
        L=[b[0] for j,b in req]
        H=[b[1] for j,b in req]
        cross=[]
        for i,j in product(range(len(f)),repeat=2):
            if f[i]==f[j]:
                continue
            for left,right in product([L[i],H[i]],[L[j],H[j]]):
                x=(right-left)/(f[i]-f[j])
                if lo<x<hi:
                    cross.append(x)
        def integrand(r):
            start=max(l+s*r for l,s in zip(L,f))
            end=min(h+s*r for h,s in zip(H,f))
            return max(0.,np.exp(c*end-.5*r*r/v)-np.exp(c*start-.5*r*r/v))/(c*norm)
        return quad(integrand,lo,hi,points=sorted(set(cross)),epsabs=4e-12)[0]

    def rates(bands=(),bridge=None,tag=False):
        cuts=[(j,window) for j,window in enumerate(e["shells"])]
        joint=measure([*cuts,*bands],bridge,tag)
        return np.array([max(0.,measure([cut,*bands],bridge,tag)-joint) for cut in cuts]+[joint])

    def counts(rate,need):
        nx,ny=need
        if nx<0 or ny<0:
            return 0.
        singles=[[x**n/factorial(n) for n in range(max(nx,ny)+1)] for x in rate]
        return np.exp(-sum(rate))*sum(singles[0][nx-i]*singles[1][ny-i]*singles[2][i]
                                     for i in range(min(nx,ny)+1))

    base=rates()
    count=np.asarray(e["counts"])
    prob=counts(base,count)
    if e["observable"]=="herald_rate":
        return float(prob)
    young=rates(tag=True)
    bands=[[(j,s),(j,m)] for j,(s,m) in enumerate(zip(e["shells"],e["marks"]))]
    mom=[rates(bands[0]+bands[1],tag=True)] if e["same_site"] else [
            rates(b,tag=True) for b in bands]
    n=len(mom)
    factor=float(np.prod(count))
    more=e["outside"]
    if more is not None:
        j=more["setting"]
        mom.append(rates([(j,e["shells"][j]),(j,more["mark"])],more["bridge"]))
        factor*=count[j]
    amplitudes=[]
    for inds in product(range(3),repeat=len(mom)):
        amplitude=np.prod([m[i] for i,m in zip(inds,mom)])
        if amplitude:
            amplitudes.append((amplitude,count-np.sum([MASK[i] for i in inds],axis=0)))

    def profile(t):
        scale=np.exp(-c*t)
        mean=base+young*scale
        return sum(m*counts(mean,k) for m,k in amplitudes)*scale**(n-1)

    return float(quad(profile,*e["parent_window"],epsabs=2e-12)[0]/(
        prob*factor*np.sqrt(2*np.pi*v0)))


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.51+.06*(i%12),.48+.05*(i%15)],
                 ratio=.45,inverse_temperature=float(x)) for i,x in enumerate(np.linspace(.081,.339,93))]


def case(**kw):
    e=dict(observable="sibling_record",ratio=.46,variances=[.78,.91],counts=[2,3],
           fields=[-.43,.58],shells=[[-.95,.64],[-.36,1.03]],
           marks=[[-.38,.51],[-.23,.47]],parent_window=[-.75,1.11],
           same_site=False,outside=None)
    e.update(kw)
    return e


def herald(e):
    return dict(observable="herald_rate",**{k:e[k] for k in
                ("ratio","variances","counts","fields","shells")})


def hidden_inputs():
    rng=np.random.default_rng(145259)
    other=dict(setting=1,mark=[-.24,.85],bridge=[.12,1.01])
    groups={"joint_heralds":[herald(case()),herald(case(counts=[1,4])),herald(case(fields=[0,0]))],
            "point_identity":[case(),case(same_site=True),case(counts=[1,1],same_site=True)],
            "outside_atom":[case(outside=other),case(same_site=True,outside=other)],
            "warm_calibration":calibration_inputs()[::11]}
    for i in range(11):
        e=case(ratio=float(rng.uniform(.32,.69)),variances=rng.uniform(.52,1.2,2).tolist(),
               fields=rng.uniform(-.9,.9,2).tolist(),counts=[1+i%4,1+(2*i+1)%4],
               marks=[[-.68,.24+i*.045],[-.25,.37+i*.065]],
               parent_window=[-1.44+.07*i,.96+.064*i],same_site=i%2==0)
        groups["joint_heralds"].append(herald(e))
        groups["point_identity"].append(e)
        groups["outside_atom"].append(dict(e,outside=dict(setting=i%2,mark=[-.39,.61],
                                                          bridge=[-.15+.05*i,.73+.09*i])))
    return groups


def checks():
    a=case(counts=[1,1],fields=[0,0],same_site=True)
    b=dict(a,same_site=False,marks=[[-.21,.38],[-.21,.38]])
    x,y=predict([a,b],TRUE_PARAMETER)
    assert x>0 and y==0
    return dict(shared_atom=float(x),distinct_atoms_impossible=float(y))
