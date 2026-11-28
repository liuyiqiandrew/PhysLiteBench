import numpy as np
from scipy.special import expit

PARAMETER="gain"
TRUE_PARAMETER=.966
CALIBRATION_SEED=28213132
NOISE_FRACTION=.0027


def staircase(e):
    V=np.r_[0.,np.cumsum(e["variances"])]
    S=np.r_[0.,np.cumsum(e["entropies"])]
    first=0
    while first<len(S)-1:
        ray=(S[first+1:]-S[first])/(V[first+1:]-V[first])
        last=first+1+int(np.flatnonzero(ray<=ray.min()+1e-13)[-1])
        yield first,last,ray.min()
        first=last


def read(e,g):
    beta=e["inverse_temperature"]
    if e["observable"]=="energy_density":
        return -beta*np.sum(e["variances"])*g*g
    rows=e["record"]
    blocks=list(staircase(e))
    selected=[b for _,b,_ in blocks]
    probability=1.
    filter_rows=e["mark_filter"]
    for i,path in enumerate(rows):
        relation=np.array([next((l for l,(x,y) in enumerate(zip(path,old)) if x!=y),len(path))
                           for old in rows[:i]],dtype=int)
        if any(level and level not in selected for level in relation):
            return 0.
        earlier=0.
        for p,j,r in blocks:
            exponent=np.sqrt(2*r)/(beta*g)
            hits=np.flatnonzero(relation>=p)
            matches=np.flatnonzero(relation>=j)
            if len(hits):
                if len(matches):
                    mass=len(matches)-exponent
                else:
                    occupied=len({tuple(rows[k][:j]) for k in hits})
                    mass=occupied*exponent-earlier
                probability*=mass/(len(hits)-earlier)
            for k in range(p,j):
                sign=filter_rows[i][k]
                if sign:
                    old={filter_rows[m][k] for m in matches}-{0}
                    if old and old!={sign}:
                        return 0.
                    if not old:
                        probability*=expit(2*beta*exponent*e["fields"][k]*sign)
            earlier=exponent
    return float(probability)


def predict(experiments,g):
    return np.array([read(e,g) for e in experiments])


def calibration_inputs():
    return [dict(observable="energy_density",entropies=[.62,.64,.77,.89],
            variances=[.28,.37,.26+.031*(i%9),.40],fields=[0.]*4,inverse_temperature=.17+.021*i)
            for i in range(27)]


def setup(i):
    ratios=[[.89,.30,1.31,1.78],[.29,1.36,.52,1.82],[.33,.82,1.65,.60],
            [.26,.78,1.95,1.21],[.39,.36,.18,1.72]][i]
    variances=[[.26,.39,.35,.21],[.31,.28,.39,.21],[.31,.36,.17,.28],
               [.31,.32,.20,.32],[.25,.43,.26,.29]][i]
    return dict(observable="marked_record",variances=variances,
                entropies=[x*y for x,y in zip(variances,ratios)],inverse_temperature=3.66+.26*i,
                fields=[[.91,-.61,.46,-.58],[-.47,.68,-.89,.57],[.78,-.52,-.45,.81],
                        [-.73,.54,.48,-.57],[.62,-.79,.51,.69]][i],
                record=[[0,0,0,0]],mark_filter=[[0,0,0,0]])


def prefix(length):
    return [0]*length+[1]*(4-length)


def hidden_inputs():
    out={k:[] for k in ("merged_depth_mark_tilts","marked_overlap_records",
                         "several_draws_on_a_single_merged_branch","energy_and_conflicts")}
    for i in range(5):
        e=setup(i)
        for signs in ([1,0,0,0],[-1,1,0,-1],[1,-1,-1,1],[0,1,-1,1]):
            out["merged_depth_mark_tilts"].append(dict(e,mark_filter=[signs]))
        for depth in (1,2,3,4):
            out["marked_overlap_records"].append(dict(e,record=[prefix(4),prefix(depth)],
                mark_filter=[[1,0,-1,1],[0,1,0,0]]))
        for depth in [j for _,j,_ in staircase(e)]:
            out["several_draws_on_a_single_merged_branch"].append(dict(e,
                record=[prefix(4),prefix(4),prefix(depth)],
                mark_filter=[[1,0,0,0],[0,-1,-1,0],[0,0,0,1]]))
        out["energy_and_conflicts"] += [calibration_inputs()[4*i],
                dict(e,record=[prefix(4),prefix(4)],mark_filter=[[0,0,1,0],[0,0,-1,0]])]
    return out


def checks():
    e=setup(0)
    laws=dict((end,np.sqrt(2*rate)/(e["inverse_temperature"]*TRUE_PARAMETER))
              for _,end,rate in staircase(e))
    J=list(laws)
    diff=np.diff(np.r_[0.,list(laws.values()),1.])
    actual=[read(dict(e,record=[prefix(4),prefix(k)],mark_filter=[[0]*4,[0]*4]),
                 TRUE_PARAMETER) for k in [0,*J]]
    assert max(abs(diff-actual))<2e-12 and abs(sum(actual)-1)<2e-12
    assert read(dict(e,record=[prefix(4),prefix(1)],mark_filter=[[0]*4,[0]*4]),TRUE_PARAMETER)==0.
    e=dict(e,record=[prefix(4),prefix(2)],mark_filter=[[1,0,0,0],[0,-1,0,0]])
    fac=np.prod(expit(2*laws[2]*e["inverse_temperature"]*np.array(e["fields"])[:2]*[1.,-1.]))
    expected=(laws[3]-laws[2])*fac
    error=abs(read(e,TRUE_PARAMETER)-expected)
    assert error<2e-12,error
    return dict(unmarked_pair_probability_total=sum(actual),marked_pair_factorization_error=error)
