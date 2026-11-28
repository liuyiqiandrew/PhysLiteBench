import numpy as np
from scipy.special import expit

PARAMETER = "gain"
TRUE_PARAMETER = .959
CALIBRATION_SEED = 28213032
NOISE_FRACTION = .0027


def read(e,gain):
    beta=e["inverse_temperature"]
    if e["observable"]=="energy_density":
        return -np.sum(e["variances"])*beta*gain**2
    ratio=np.sqrt(2*np.array(e["entropies"])/e["variances"])/(gain*beta)
    mark=expit(2*ratio*beta*e["fields"])
    seen=[{} for _ in ratio]
    rows=e["record"]
    out=1.
    for i,path in enumerate(rows):
        pa=0.
        for k,a in enumerate(ratio):
            cohort=[x for x in rows[:i] if x[:k]==path[:k]]
            if cohort:
                n=sum(x[k]==path[k] for x in cohort)
                if n:
                    probability=(n-a)/(len(cohort)-pa)
                else:
                    probability=(len({x[k] for x in cohort})*a-pa)/(len(cohort)-pa)
                out*=probability
            branch=tuple(path[:k+1])
            request=e["mark_filter"][i][k]
            if request:
                prev=seen[k].get(branch)
                if prev is not None:
                    if prev!=request:
                        return 0.
                else:
                    seen[k][branch]=request
                    out*=mark[k] if request==1 else 1-mark[k]
            pa=a
    return float(out)


def predict(experiments,gain):
    return np.array([read(e,gain) for e in experiments])


def calibration_inputs():
    return [dict(observable="energy_density",fields=[0.,0.,0.],entropies=[.6,.91,.81],
                 variances=[.54,.43,.26],inverse_temperature=.13+.027*i) for i in range(26)]


def setup(i):
    vs=[[.44,.29,.37],[.32,.44,.37],[.22,.37,.46],[.37,.18,.34],[.29,.31,.24]][i]
    sl=[[.33,.79,1.24],[.39,.69,1.83],[.28,.71,1.34],[.48,1.16,1.87],[.36,.81,1.53]][i]
    return dict(observable="marked_record",inverse_temperature=2.52+.36*i,
                variances=vs,entropies=[r*v for r,v in zip(sl,vs)],
                fields=[[.33,-.58,.45],[-.49,.78,.61],[.51,-.39,.84],[.71,.46,-.7],[-.55,-.41,-.73]][i],
                record=[[0,0,0]],mark_filter=[[0,0,0]])


def hidden_inputs():
    groups={k:[] for k in ("branch_field_response","nested_marked_replicas",
                            "disjoint_and_contradictory_records","calibration_and_no_filter")}
    for i in range(5):
        e=setup(i)
        for s in ([1,0,0],[0,-1,1],[-1,1,-1],[1,1,1]):
            groups["branch_field_response"].append(dict(e,mark_filter=[s]))
        groups["nested_marked_replicas"] += [
            dict(e,record=[[0,0,0],[0,0,1],[0,2,5]],mark_filter=[[0,1,0],[0,1,-1],[0,0,1]]),
            dict(e,record=[[0,0,0],[0,0,0],[0,1,2]],mark_filter=[[1,0,-1],[0,1,-1],[1,0,0]]),
            dict(e,record=[[0,0,0],[0,1,1],[0,1,1]],mark_filter=[[0,-1,1],[1,0,0],[1,-1,1]])]
        groups["disjoint_and_contradictory_records"] += [
            dict(e,record=[[0,0,0],[0,0,0]],mark_filter=[[1,0,0],[-1,0,0]]),
            dict(e,record=[[0,0,0],[1,0,0]],mark_filter=[[1,-1,0],[1,1,-1]]),
            dict(e,record=[[0,0,0],[0,2,0]],mark_filter=[[0,-1,0],[1,-1,-1]])]
        groups["calibration_and_no_filter"] += [calibration_inputs()[4*i],dict(e)]
    return groups


def checks():
    e=setup(2)
    a=np.sqrt(2*np.array(e["entropies"])/e["variances"])/(TRUE_PARAMETER*e["inverse_temperature"])
    pair=[read(dict(e,record=[[0,0,0],[0]*i+[1]*(3-i)],mark_filter=[[0,0,0],[0,0,0]]),
               TRUE_PARAMETER) for i in range(4)]
    expected=np.diff(np.r_[0.,a,1.])
    error=max(abs(np.array(pair)-expected))
    assert error<1e-12 and abs(sum(pair)-1)<1e-12,error
    scenario=dict(e,mark_filter=[[1,-1,1]])
    numerator=read(scenario,TRUE_PARAMETER)
    flipped=dict(scenario,fields=(-np.array(e["fields"])).tolist(),mark_filter=[[-1,1,-1]])
    assert abs(read(flipped,TRUE_PARAMETER)-numerator)<1e-14
    rng=np.random.default_rng(8207)
    alpha=.38
    shift=.94
    estimates=[]
    for _ in range(20):
        arrivals=np.cumsum(rng.exponential(size=(1500,260)),axis=1)
        M=2*rng.integers(0,2,size=arrivals.shape)-1
        W=np.exp(shift*M)*(arrivals[:,0,None]/arrivals)**(1/alpha)
        estimates.append(np.mean(np.sum(W*(M>0),axis=1)/W.sum(axis=1)))
    delta=abs(np.mean(estimates)-expit(2*alpha*shift))
    assert delta<.008,delta
    return dict(all_pair_probabilities=sum(pair),pair_weights_error=error,mark_tilt_mc_error=float(delta))
