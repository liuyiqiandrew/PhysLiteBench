from collections import Counter,defaultdict

import numpy as np
from scipy.special import gamma,roots_jacobi

PARAMETER = "gain"
TRUE_PARAMETER = 1.016
CALIBRATION_SEED = 28213717
NOISE_FRACTION = .0025


def base(e,g):
    return np.sqrt(2*np.asarray(e["entropies"])/e["variances"])/(e["inverse_temperature"]*g)


def partition(d,t,C):
    z=1.
    n=0
    for i,m in enumerate(C):
        if i:
            z*=t+d*i
        z*=gamma(m-d)/gamma(1-d)
        n+=m
    return z*gamma(1+t)/gamma(n+t)


def evaluate(e,g,order=37):
    beta=e["inverse_temperature"]
    if e["observable"]=="warm_energy":
        return -beta*g*g*sum(e["variances"])
    a,b=base(e,g)
    record=[tuple(x) for x in e["record"]]
    sites=defaultdict(Counter)
    for i,j in record:
        sites[i][j]+=1
    family=Counter(i for i,j in record)
    fixed_regions=sorted({i for i,j in record[:-1]},reverse=True)
    edits=[tuple(beta*x for x in pair) for pair in e["edits"]]
    unedited=partition(a,0,family.values())
    for C in sites.values():
        unedited*=partition(b,-a,C.values())
    blocks=[(a,0,family,fixed_regions)]
    for i in fixed_regions:
        fs=sorted({j for I,j in record[:-1] if I==i},reverse=True)
        blocks.append((b,-a,sites[i],fs))
    all_axes=[(idx,item) for idx,(_,_,_,labels) in enumerate(blocks) for item in labels]
    dimension=len(all_axes)
    draws={}
    probs=1.
    for idx,(edge,theta,counts,visited) in enumerate(blocks):
        residual=theta+sum(counts.values())
        stick=1.
        values={}
        for label in visited:
            m=counts[label]-edge
            residual-=m
            x,w=roots_jacobi(order,residual-1,m-1)
            shape=[1]*dimension
            shape[all_axes.index((idx,label))]=order
            x=(x.reshape(shape)+1)/2
            probs=probs*(w/w.sum()).reshape(shape)
            values[label]=stick*x
            stick=stick*(1-x)
        draws[idx]=values
    P=draws[0]
    Q={i:draws[1+k] for k,i in enumerate(fixed_regions)}
    touched=list(dict.fromkeys(record[:-1]))
    region_sites=defaultdict(set)
    for i,j in touched:
        region_sites[i].add(j)
    likelihood=1.
    active_f=defaultdict(float)
    active_s=defaultdict(float)
    for location,edit,next_location in zip(record,edits,record[1:]):
        i,j=location
        active_f[i] += edit[0]
        active_s[i,j] += edit[1]
        norm=1-sum(P.values())
        for i,all_j in region_sites.items():
            internally_untouched=1-sum(Q[i].values())
            modified=internally_untouched
            for j in all_j:
                modified=modified+Q[i][j]*np.exp(active_s[i,j])
            norm=norm+P[i]*np.exp(active_f[i])*modified
        nx,ny=next_location
        likelihood=likelihood*np.exp(active_f[nx]+active_s[nx,ny])/norm
    return float(np.sum(probs*likelihood)*unedited)


def predict(experiments,g):
    return np.array([evaluate(e,g) for e in experiments])


def calibration_inputs():
    return [dict(observable="warm_energy",entropies=[.15+.009*(i%10),.58],
                 variances=[.56+.021*i,.51],inverse_temperature=.082+.01*i) for i in range(31)]


def setup(i):
    return dict(observable="feedback_record",entropies=[.121+.040*i,.45+.046*i],
                variances=[.75-.029*i,.64+.053*i],inverse_temperature=3.25+.265*i,
                edits=[[.41-.16*i,-.51+.19*i],[-.32+.19*i,.44-.16*i]],record=[[0,0],[1,0],[0,0]])


def hidden_inputs():
    G={"addressed_site_recurrence":[],"changed_family_switching":[],
       "shared_normalization_for_two_edits":[],"zero_feedback_and_calibration":[]}
    for i in range(5):
        e=setup(i)
        G["addressed_site_recurrence"] += [dict(e,record=r) for r in
            ([[0,0],[0,0],[0,0]],[[0,0],[0,1],[0,0]],[[0,0],[0,1],[0,1]])]
        G["changed_family_switching"] += [dict(e,record=r) for r in
            ([[0,0],[1,0],[0,0]],[[0,0],[1,0],[1,1]],[[0,0],[1,0],[2,0]])]
        G["shared_normalization_for_two_edits"] += [
            dict(e,edits=[[.61,-.54],[-.55,.52]],record=[[0,0],[0,1],[0,0]]),
            dict(e,edits=[[-.61,.56],[.52,-.58]],record=[[0,0],[1,0],[0,0]]),
            dict(e,edits=[[.54,-.67],[.53,.59]],record=[[0,0],[0,0],[1,0]])]
        G["zero_feedback_and_calibration"] += [
            dict(e,edits=[e["edits"][0]],record=[[0,0],[0,1]]),
            dict(e,edits=[[0.,0.],[0.,0.]]),calibration_inputs()[4*i]]
    return G


def checks():
    g=TRUE_PARAMETER
    e=dict(setup(1),record=[[0,0],[0,1]],edits=[[.42,-.53]])
    p=evaluate(e,g)
    a,b=base(e,g)
    beta=e["inverse_temperature"]
    rng=np.random.default_rng(235911)
    x=rng.beta(1-a,a,size=120000)
    y=rng.beta(1-b,b-a,size=len(x))
    u,v=np.exp(beta*np.asarray(e["edits"][0]))
    tx=u*x*(1-y)/(1-x+u*x*(1-y+v*y))
    err=tx.std()/np.sqrt(len(tx))
    assert abs(tx.mean()-p)<5*err+.00005
    normalized=sum(evaluate(dict(e,record=[[0,0],r]),g) for r in ([0,0],[0,1],[1,0]))
    assert abs(normalized-1)<2e-10
    difficult=hidden_inputs()["shared_normalization_for_two_edits"][-2]
    difference=abs(evaluate(difficult,g,45)-evaluate(difficult,g))
    assert difference<3e-7,difference
    return dict(feedback_probability=p, simulated_probability=float(tx.mean()),
                standard_error=float(err),two_draw_normalization=normalized,
                independent_quadrature_refinement=difference)
