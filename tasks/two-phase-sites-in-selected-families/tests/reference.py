from collections import defaultdict
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

PARAMETER="gain"
TRUE_PARAMETER=.958
CALIBRATION_SEED=28213558
NOISE_FRACTION=.0026


def data(e,g):
    s=np.asarray(e["entropies"])
    v=np.asarray(e["variances"])
    critical=np.sqrt(2*s/v)/g
    a,b0,b1=critical/e["inverse_temperature"]
    c=np.exp(-critical[1:]*e["offsets"])/np.sqrt(4*np.pi*s[1:])
    return a if e["protocol"]=="thermal" else 0.,np.array([b0,b1]),c


@lru_cache(maxsize=1800)
def function_integral(b,C,power):
    n=min(b)
    j=b.index(n)
    k=1-j
    r=b[k]/n
    exponent=power/n
    z=C[k]*C[j]**(-r)
    def f(t):
        u=t**(1/exponent)
        return np.exp(-u-z*u**r)/exponent
    radial=quad(f,0.,np.inf,epsabs=3e-11,epsrel=2e-9,limit=175)[0]
    return radial/(n*C[j]**exponent)


def evaluate(e,g):
    if e["observable"]=="bare_energy_density":
        v=e["variances"]
        return -e["inverse_temperature"]*g*g*(v[0]+v[1+e["phase"]])
    a,b,c=data(e,g)
    C=c*gamma(1-b)

    def M(power):
        return function_integral(tuple(b),tuple(C),float(power))

    denominator=sum(C[j]*b[j]*M(b[j]-a) for j in (0,1))
    # Integration by parts in the fractional-moment formula is equivalent
    # to summing the two phase probabilities from a single unrestricted draw.
    denominator*=gamma(len(e["record"])-a)/gamma(1-a)
    groups=defaultdict(list)
    for i,k in e["record"]:
        groups[i].append(k)
    marked={}
    possibilities=[]
    for key,marks in groups.items():
        valid=[r for r in (0,1) if all(m in (-1,r) for m in marks)]
        possibilities.append(valid)
        marked[key]={r:b[r]*c[r]*gamma(len(marks)-b[r]) for r in valid}
    value=0.
    for choice in product(*possibilities):
        power=sum(b[r] for r in choice)-a
        mark=np.prod([marked[key][r] for key,r in zip(groups,choice)])
        value+=M(power)*mark
    return float(value/denominator)


def predict(experiments,g):
    return np.array([evaluate(e,g) for e in experiments])


def calibration_inputs():
    return [dict(observable="bare_energy_density",phase=i%2,entropies=[.13+.01*(i%8),.52,.45],
                 variances=[.57+.018*(i%9),.38+.013*i,.49],
                 inverse_temperature=.086+.012*i) for i in range(31)]


def setup(i):
    return dict(observable="site_record",entropies=[[.061,.41,.51],[.38,.33,.59],[.42,.37,.60],
               [.24,.31,.63],[.29,.67,.29]][i],
                variances=[[.75,.39,.53],[.55,.45,.38],[.58,.47,.34],
                           [.59,.51,.39],[.72,.43,.67]][i],
                inverse_temperature=3.12+.19*i,offsets=[.25-.19*i,-.29+.17*i],
                protocol="thermal",record=[[0,-1]])


def hidden_inputs():
    g={k:[] for k in ("phase_choice_after_family_selection","within_phase_coincidences",
                      "mixed_phase_site_records","same_exponent_and_calibration")}
    for i in range(5):
        e=setup(i)
        for phase in (0,1):
            for protocol in ("thermal","uniform"):
                g["phase_choice_after_family_selection"].append(dict(e,protocol=protocol,record=[[0,phase]]))
        for r in ([[0,0],[0,0]],[[0,1],[0,1]],[[0,-1],[0,-1],[0,-1]],[[0,0],[1,0],[0,0],[3,-1]]):
            g["within_phase_coincidences"].append(dict(e,record=r))
        for r in ([[0,0],[1,1]],[[0,0],[0,1]],[[0,0],[1,0],[2,1]],
                  [[0,0],[1,1],[0,0],[2,1]],[[0,-1],[0,0],[1,1],[2,1]]):
            g["mixed_phase_site_records"].append(dict(e,record=r))
        e0=dict(e,entropies=[e["entropies"][0],.47,.47],variances=[e["variances"][0],.37,.37])
        g["same_exponent_and_calibration"] += [dict(e0,record=[[0,-1],[0,-1]]),
                    dict(e0,offsets=[0.,0.],record=[[0,0]]),calibration_inputs()[4*i]]
    return g


def checks():
    e=setup(0)
    g=TRUE_PARAMETER
    a,b,c=data(e,g)
    rng=np.random.default_rng(411034)
    rows=[]
    for _ in range(24):
        atoms=[]
        sums=[]
        for β,λ in zip(b,c):
            r=np.cumsum(rng.exponential(size=(2000,190)),axis=1)
            W=(λ/r)**(1/β)
            tail=β*λ/(1-β)*W[:,-1]**(1-β)
            atoms.append(W)
            sums.append(W.sum(axis=1)+tail)
        total=sums[0]+sums[1]
        mass=total**a
        rows.append(np.stack([mass,mass*sums[0]/total,mass*np.sum(atoms[0]**2,axis=1)/total**2]))
    z=np.concatenate(rows,axis=1)
    den,*out=z.mean(axis=1)
    estimates=np.asarray(out)/den
    checks=[evaluate(dict(e,record=[[0,0]]),g),
            evaluate(dict(e,record=[[0,0],[0,0]]),g)]
    stderr=(z[1:]-estimates[:,None]*z[0]).std(axis=1)/(np.sqrt(z.shape[1])*den)
    assert np.all(abs(estimates-checks)<6*stderr+.0035),(estimates,checks,stderr)
    equal=dict(e,entropies=[.10,.51,.51],variances=[.57,.46,.46],record=[[0,-1],[0,-1]])
    aa,bb,_=data(equal,g)
    check=evaluate(equal,g)
    assert abs(check-(1-bb[0])/(1-aa))<5e-10,check
    return dict(simulated_records=estimates.tolist(),exact_probabilities=checks,
                spread=stderr.tolist(),equal_exponent_pair=check)
