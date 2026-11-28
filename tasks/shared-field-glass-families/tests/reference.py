from math import exp, lgamma

import numpy as np
from scipy.integrate import quad

PARAMETER = "gain"
TRUE_PARAMETER = .966
CALIBRATION_SEED = 28213069
NOISE_FRACTION = .0028


def read(e,gain):
    beta=e["inverse_temperature"]
    if e["observable"]=="energy_density":
        return -beta*sum(e["variances"])*gain*gain
    d1,d2=(np.sqrt(2*np.asarray(e["entropies"])/e["variances"])/(beta*gain)).tolist()
    config=e["record"]
    keep=e["root_filter"]
    stage=e["settings"]
    n,m=stage.count(0),stage.count(1)
    parents={}
    leaf_factor=1.
    for i,(p,q) in enumerate(config):
        older=parents.setdefault(p,[])
        if older:
            old_leaf={config[j][1] for j in older}
            existing=sum(config[j][1]==q for j in older)
            weight=(existing-d2) if existing else (len(old_leaf)*d2-d1)
            leaf_factor*=weight/(len(older)-d1)
        older.append(i)
    first,second=[(exp(-beta*f),exp(beta*f)) for f in e["fields"]]
    sizes=[]
    for indices in parents.values():
        signs={keep[j] for j in indices} - {0}
        if len(signs)>1:
            return 0.
        total=len(indices)
        a=sum(stage[j]==0 for j in indices)
        pref=d1*exp(lgamma(total-d1)-lgamma(1-d1))/2
        pieces=[pref*first[j]**a*second[j]**(total-a)
                if not signs or 2*j-1 in signs else 0. for j in range(2)]
        sizes.append((total,pieces))

    def moment(s,t):
        q0=s*first[0]+t*second[0]
        q1=s*first[1]+t*second[1]
        powered=(q0**d1+q1**d1)/2
        out=1.
        for k,(f0,f1) in sizes:
            out*=f0*q0**(d1-k)+f1*q1**(d1-k)
        return out/powered**len(sizes)

    factor=leaf_factor*exp(lgamma(len(sizes)))/d1
    if n==0 or m==0:
        return factor*moment(float(m==0),float(n==0))*exp(-lgamma(n+m))
    integral=quad(lambda x:exp(m*x)*moment(1.,exp(x)),-45.,45.,epsabs=4e-12,limit=100)[0]
    return factor*integral*exp(-lgamma(n)-lgamma(m))


def predict(experiments,gain):
    return np.array([read(e,gain) for e in experiments])


def calibration_inputs():
    return [dict(observable="energy_density",entropies=[.79,.81],variances=[.37+.02*(i%9),.34],
                 inverse_temperature=.16+.027*i,fields=[0.,0.]) for i in range(27)]


def setup(i):
    variance=[[.36,.52],[.72,.19],[.67,.41],[.58,.38],[.45,.55]][i]
    slopes=[[.31,1.17],[.51,1.23],[.66,1.71],[.32,.84],[.43,1.01]][i]
    return dict(observable="overlap_record",variances=variance,
                entropies=[u*v for u,v in zip(variance,slopes)],
                inverse_temperature=2.53+.34*i,fields=[-.67+.1*i,.93-.08*i],
                settings=[0,1],record=[[0,0],[0,1]],root_filter=[0,0])


def hidden_inputs():
    groups={k:[] for k in ("two_field_family_labels","joint_family_and_leaf_weights",
                            "field_disagreement_and_mark_filters","single_field_and_calibration")}
    for i in range(5):
        e=setup(i)
        groups["two_field_family_labels"].extend([
            dict(e),dict(e,record=[[0,0],[1,0]]),dict(e,root_filter=[1,1]),
            dict(e,record=[[0,0],[0,0]],root_filter=[-1,0])])
        groups["joint_family_and_leaf_weights"].extend([
            dict(e,settings=[0,1,0],record=[[0,0],[0,0],[0,1]],root_filter=[0,0,0]),
            dict(e,settings=[1,1,0],record=[[0,0],[1,0],[0,1]],root_filter=[0,-1,0]),
            dict(e,settings=[0,1,1,0],record=[[0,0],[0,1],[1,0],[1,0]],root_filter=[1,0,-1,-1])])
        for f in (.1,.51,1.03):
            groups["field_disagreement_and_mark_filters"].append(dict(e,fields=[-.38,f],
                record=[[0,0],[1,0]],root_filter=[-1,1]))
        groups["single_field_and_calibration"] += [
            calibration_inputs()[3*i],dict(e,settings=[0,0]),dict(e,settings=[1,1],root_filter=[1,0])]
    return groups


def checks():
    e=setup(0)
    a,b=np.sqrt(2*np.array(e["entropies"])/e["variances"])/(TRUE_PARAMETER*e["inverse_temperature"])
    y=read(dict(e,fields=[.38,.38]),TRUE_PARAMETER)
    assert abs(y-(b-a))<1e-13,y
    z=read(e,TRUE_PARAMETER)
    swapped=dict(e,settings=[1,0],fields=e["fields"][::-1])
    assert abs(z-read(swapped,TRUE_PARAMETER))<1e-13
    rng=np.random.default_rng(26452)
    collection=[]
    for _ in range(20):
        times=np.cumsum(rng.exponential(size=(3000,380)),axis=1)
        marks=2*rng.integers(0,2,size=times.shape)-1
        base=(times[:,0,None]/times)**(1/a)
        p0=base*np.exp(e["inverse_temperature"]*e["fields"][0]*marks)
        p1=base*np.exp(e["inverse_temperature"]*e["fields"][1]*marks)
        norm=np.sum(p0,axis=1)*np.sum(p1,axis=1)
        geometry=np.sum(p0*p1,axis=1)/norm
        collection.append(float(geometry.mean())*(b-a)/(1-a))
    discrepancy=abs(np.mean(collection)-z)
    assert discrepancy<.003,discrepancy
    return dict(single_field_same_family_different_leaf_error=abs(y-(b-a)),
                simulated_and_integrated_two_field_error=float(discrepancy))
