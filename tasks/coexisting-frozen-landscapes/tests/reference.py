from collections import Counter,defaultdict
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

PARAMETER="gain"
TRUE_PARAMETER=.984
CALIBRATION_SEED=28213259
NOISE_FRACTION=.0027


def scales(e,gain):
    exponent=np.sqrt(2*np.array(e["entropies"])/e["variances"])/(gain*e["inverse_temperature"])
    intensities=np.zeros(2)
    for phase,(a,b) in enumerate(exponent):
        if e["log_bias"][phase]==min(e["log_bias"]):
            root,tip=e["entropies"][phase]
            child=gamma(1-b)/np.sqrt(4*np.pi*tip)
            thermal_moment=child**(a/b)*gamma(1-a/b)/gamma(1-a)
            intensities[phase]=(np.exp(-a*e["inverse_temperature"]*e["offset"][phase])
                                *thermal_moment/np.sqrt(4*np.pi*root))
    return exponent,intensities


def insertion(leaves,b,a):
    counter=Counter([leaves[0]])
    p=1.
    for i,l in enumerate(leaves[1:],start=1):
        n=counter[l]
        if n:
            p*=(n-b)/(i-a)
        else:
            p*=(b*len(counter)-a)/(i-a)
        counter[l]+=1
    return p


@lru_cache(maxsize=1200)
def normalization(a,K,marks):
    # Use the active phase with the greatest Laplace exponent as the radial variable.
    pivot=max((i for i in range(2) if K[i]>0),key=lambda i:a[i])
    q=1-pivot
    depth=(a[0]*marks[0]+a[1]*marks[1])/a[pivot]
    c=K[q]/K[pivot]**(a[q]/a[pivot])
    # Absorb the radial Gamma singularity with the power transform u=w^(1/depth).
    def kernel(w):
        u=w**(1/depth)
        return np.exp(-u-c*u**(a[q]/a[pivot]))/depth
    value=quad(kernel,0.,np.inf,epsabs=2e-12,epsrel=3e-11,limit=250)[0]
    return value/(a[pivot]*K[pivot]**depth)


def event(e,gain):
    beta=e["inverse_temperature"]
    if e["observable"]=="bare_energy_density":
        return -beta*gain**2*sum(e["variances"][e["phase"]])
    ex,c=scales(e,gain)
    a,b=ex.T
    K=c*gamma(1-a)
    grouped=defaultdict(list)
    for family,leaf,phase in e["record"]:
        grouped[family].append((leaf,phase))
    choices=[]
    for rows in grouped.values():
        labels=[entry[0] for entry in rows]
        row=[]
        for k in range(2):
            if c[k]>0 and all(p in (-1,k) for _,p in rows):
                factor=insertion(labels,b[k],a[k])*c[k]*a[k]*gamma(len(rows)-a[k])
                row.append((k,factor))
        if not row:
            return 0.
        choices.append(row)
    p=0.
    for combination in product(*choices):
        phases,factors=zip(*combination)
        n1=sum(phases)
        radial=normalization(tuple(a),tuple(K),(len(phases)-n1,n1))
        p+=np.prod(factors)*radial
    return p/gamma(len(e["record"]))


def predict(experiments,gain):
    return np.asarray([event(e,gain) for e in experiments])


def calibration_inputs():
    return [dict(observable="bare_energy_density",phase=i%2,entropies=[[.17,.55],[.21,.46]],
                 variances=[[.56+.021*(i%7),.41],[.45,.31+.014*i]],inverse_temperature=.12+.014*i)
            for i in range(30)]


def setup(i):
    return dict(observable="overlap_record",entropies=[[.17+.026*i,.52-.043*i],[.32-.051*i,.42+.055*i]],
                variances=[[.64-.033*i,.39+.021*i],[.57+.016*i,.34+.038*i]],
                inverse_temperature=2.81+.24*i,log_bias=[0.,0.],
                offset=[-.42+.18*i,.24-.13*i],record=[[0,0,-1]])


def hidden_inputs():
    g={k:[] for k in ("phase_coexistence","phase_and_nested_overlap",
                       "common_normalizer","logarithmic_selection_and_calibration")}
    for i in range(5):
        e=setup(i)
        for h in (e["offset"],[0.,0.],[-.24,.36],[.47,-.48]):
            g["phase_coexistence"].append(dict(e,offset=h,record=[[0,0,0]]))
        size=2.4+1.05*i
        same_slopes=dict(e,entropies=[e["entropies"][0],[size*x for x in e["entropies"][0]]],
                         variances=[e["variances"][0],[size*x for x in e["variances"][0]]],
                         offset=[0.,0.])
        g["phase_coexistence"].append(dict(same_slopes,record=[[0,0,0]]))
        g["phase_and_nested_overlap"].append(dict(same_slopes,record=[[0,0,0],[0,0,0]]))
        for rows in ([[0,0,0],[0,0,0]],[[0,0,1],[0,1,1]],[[0,0,0],[1,1,0]],
                     [[0,0,-1],[0,0,-1],[0,1,0],[1,0,1]],[[0,0,1],[0,0,0]]):
            g["phase_and_nested_overlap"].append(dict(e,record=rows))
        for rows in ([[0,0,0],[1,0,1]],[[0,0,0],[1,0,-1],[1,0,1]],
                     [[0,0,-1],[1,0,-1],[2,0,-1]],[[0,0,-1],[0,0,-1],[0,1,-1]]):
            g["common_normalizer"].append(dict(e,record=rows))
        g["logarithmic_selection_and_calibration"] += [
            dict(e,log_bias=[0.,.27],record=[[0,0,1]]),
            dict(e,log_bias=[.13,0.],record=[[0,0,-1],[0,0,1]]),
            dict(e,log_bias=[.13,0.],record=[[0,0,-1],[0,1,1]]),
            calibration_inputs()[3*i]]
    return g


def checks():
    e=setup(0)
    ex,coefficients=scales(e,TRUE_PARAMETER)
    rng=np.random.default_rng(55309)
    n=37000
    atom=[]
    total=[]
    for a,c in zip(ex[:,0],coefficients):
        arrivals=np.cumsum(rng.exponential(size=(n,110)),axis=1)
        x=(c/arrivals)**(1/a)
        remainder=a*c**(1/a)*arrivals[:,-1]**(1-1/a)/(1-a)
        total.append(x.sum(axis=1)+remainder)
        atom.append(x)
    z=total[0]+total[1]
    first=total[0]/z
    same=(atom[0]**2).sum(axis=1)/z**2
    first_leaf=(1-ex[0,1])/(1-ex[0,0])*same
    actual=np.stack([first,same,first_leaf])
    average=actual.mean(axis=1)
    deviations=actual.std(axis=1)/np.sqrt(n)
    p=event(dict(e,record=[[0,0,0]]),TRUE_PARAMETER)
    leaf=event(dict(e,record=[[0,0,0],[0,0,0]]),TRUE_PARAMETER)
    other=event(dict(e,record=[[0,0,0],[0,1,0]]),TRUE_PARAMETER)
    expected=np.array([p,leaf+other,leaf])
    assert np.all(abs(average-expected)<5*deviations+.001),(expected,average,deviations)
    samephase=dict(e,offset=[0.,0.],entropies=[e["entropies"][0]]*2,variances=[e["variances"][0]]*2,
                   record=[[0,0,0]])
    symmetry=abs(event(samephase,TRUE_PARAMETER)-.5)
    assert symmetry<1e-10,symmetry
    dominating=dict(e,log_bias=[.4,.0],record=[[0,0,1],[0,1,1]])
    dominance=abs(event(dominating,TRUE_PARAMETER)-(ex[1,1]-ex[1,0]))
    assert dominance<1e-10,dominance
    return dict(Poisson_jump_predictions=average.tolist(),Poisson_standard_errors=deviations.tolist(),
                exact_probabilities=expected.tolist(),identical_phase_symmetry_error=symmetry,
                logarithmic_dominance_error=dominance)
