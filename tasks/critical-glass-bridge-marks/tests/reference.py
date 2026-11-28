from collections import Counter
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_ndtr,ndtri
from scipy.stats import norm,qmc

PARAMETER="gain"
TRUE_PARAMETER=.946
CALIBRATION_SEED=28213229
NOISE_FRACTION=.0026
X,W=leggauss(110)


def scale(e,gain):
    beta0=np.sqrt(2*e["root_entropy"]/e["variances"][0])/gain
    beta1=np.sqrt(2*e["inner_ratio"])/gain
    v=np.asarray(e["variances"][1:])
    transform=np.array([[1.,0.,0.],[1.,1.,0.]])-(np.cumsum(v)[:2]/v.sum())[:,None]
    covariance=gain**2*(transform*v)@transform.T
    shifted=covariance@(beta1*np.asarray(e["field"]))
    bound=-np.cumsum(e["detuning"])[:2]/beta1
    return beta0,beta1,covariance,shifted,bound


@lru_cache(maxsize=2500)
def integration(mean,cov,lo,hi):
    # Integrate the second residual, then use the exact conditional interval mass.
    sd1=np.sqrt(cov[2])
    sd0=np.sqrt(cov[0]-cov[1]**2/cov[2])
    limits=np.clip((np.array([lo[1],hi[1]])-mean[1])/sd1,-11,11)
    length=limits[1]-limits[0]
    if length<=0 or lo[0]>=hi[0]:
        return 0.
    z=(limits.sum()+length*X)/2
    m0=mean[0]+z*cov[1]/sd1
    mass=norm.cdf((hi[0]-m0)/sd0)-norm.cdf((lo[0]-m0)/sd0)
    return float(length/2*(W@(norm.pdf(z)*mass)))


def insert(labels,alpha,theta):
    occupied=Counter([labels[0]])
    p=1.
    for n,label in enumerate(labels[1:],start=1):
        factor=(occupied[label]-alpha if occupied[label]
                else theta+len(occupied)*alpha)
        p*=factor/(theta+n)
        occupied[label]+=1
    return p


def evaluate(e,gain):
    if e["observable"]=="energy_density":
        return -e["inverse_temperature"]*gain**2*sum(e["variances"])
    beta=e["inverse_temperature"]
    beta0,beta1,C,mean,lower=scale(e,gain)
    cov=(C[0,0],C[0,1],C[1,1])
    U=(np.inf,np.inf)
    normalizer=integration(tuple(mean),cov,tuple(lower),U)
    rows=e["record"]
    identities=[tuple(r["path"]) for r in rows]
    roots=[d[0] for d in identities]
    output=insert(roots,beta0/beta,0.)
    for root in set(roots):
        children=[label[1:] for label in identities if label[0]==root]
        if len({c[0] for c in children})!=len(set(children)):
            return 0.
        output*=insert(children,beta1/beta,-beta0/beta)
    for tip in set(identities):
        lo,hi=lower.copy(),np.full(2,np.inf)
        for row in rows:
            if tuple(row["path"])==tip:
                for k in (0,1):
                    if row["lower"][k] is not None:
                        lo[k]=max(lo[k],row["lower"][k])
                    if row["upper"][k] is not None:
                        hi[k]=min(hi[k],row["upper"][k])
        output*=integration(tuple(mean),cov,tuple(lo),tuple(hi))/normalizer
    return output


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments])


def calibration_inputs():
    return [dict(observable="energy_density",variances=[.36+.013*(i%8),.25,.31,.46],
                 root_entropy=.13+.003*i,inner_ratio=.61,
                 detuning=[.24,-.62,.38],inverse_temperature=.10+.016*i)
            for i in range(31)]


def row(path,lo=None,hi=None):
    return dict(path=path,lower=lo or [None,None],upper=hi or [None,None])


def setup(i):
    a,b=[[.42,.37],[-.20,.58],[.37,-.63],[-.51,.30],[.11,-.48]][i]
    return dict(observable="bridge_record",variances=[.60-.05*i,.28+.038*i,.72-.10*i,.34+.026*i],
                root_entropy=.101+.01*i,inner_ratio=.49+.051*i,detuning=[a,b,-a-b],
                inverse_temperature=2.60+.23*i,
                field=[-.65+.33*i,.49-.25*i],record=[row([0,0,0,0])])


def hidden_inputs():
    g={key:[] for key in ("rectangles_at_ancestor_bounds","field_reweighted_residuals",
                          "coincidence_with_bridge_selection","closed_branch_and_calibration")}
    p,q,r,s=[0,1,2,3],[0,5,2,3],[1,1,2,3],[0,1,7,3]
    for i in range(5):
        e=setup(i)
        for lo,hi in (([-.32,-.37],[.52,.48]),([None,None],[.26,.21]),
                      ([.02,.16],[None,.74]),([-.42,.22],[.67,None])):
            g["rectangles_at_ancestor_bounds"].append(dict(e,record=[row(p,lo,hi)]))
        for field in ([0.,0.],[.78,-.47],[-.61,.81]):
            g["field_reweighted_residuals"].append(dict(e,field=field,
                    record=[row(p,[-.19,.11],[.61,None])]))
        for group in ([row(p,[.14,None]),row(p,[None,.13],[.93,.81])],
                      [row(p,[.05,.10]),row(q,[None,-.25],[.73,None]),row(p)],
                      [row(p,[None,None],[.77,.61]),row(q,[.18,.11]),row(r,[.34,None])],
                      [row(p),row(q),row(p),row(q),row(p,[.27,None])]):
            g["coincidence_with_bridge_selection"].append(dict(e,record=group))
        g["closed_branch_and_calibration"] += [
            dict(e,record=[row(p),row(s)]),dict(e,record=[row(p),row(p),row(q),row(r)]),
            dict(e,record=[row(p),row(q),row(r)]),calibration_inputs()[4*i]]
    return g


def checks():
    e=setup(0)
    root,beta,cov,mu,lo=scale(e,TRUE_PARAMETER)
    u=qmc.Sobol(2,scramble=True,seed=66241).random_base2(18)
    rng=ndtri(u)@np.linalg.cholesky(cov).T
    weights=np.exp(rng@(beta*np.asarray(e["field"])))
    allowed=np.all(rng>=lo,axis=1)
    box=(rng[:,0]<=.45)&(rng[:,1]>=.15)
    qmc_value=float(weights[allowed&box].sum()/weights[allowed].sum())
    reference=evaluate(dict(e,record=[row([0,0,0,0],[None,.15],[.45,None])]),TRUE_PARAMETER)
    assert abs(qmc_value-reference)<3e-4,(qmc_value,reference)

    widths=np.asarray(e["variances"][1:])
    ancestry=np.cumsum(widths)[:2]
    full=widths.sum()
    future=full-ancestry
    x=lo+np.array([.49,.38])
    powers=[]
    for N in (900.,40000.,1000000.):
        prefix=-N*beta*TRUE_PARAMETER**2*ancestry+np.sqrt(N)*x
        full_extreme=-N*beta*TRUE_PARAMETER**2*full+np.log(N)/(2*beta)
        possible_siblings=N*e["inner_ratio"]*future-np.sqrt(N)*np.cumsum(e["detuning"])[:2]
        possible_siblings+=log_ndtr((full_extreme-prefix)/(TRUE_PARAMETER*np.sqrt(N*future)))
        powers.append((possible_siblings/np.sqrt(N)).tolist())
    expected=(-beta*(x-lo)).tolist()
    assert max(abs(np.asarray(powers[-1])-expected))<.004,(powers,expected)
    collision=evaluate(dict(e,record=[row([0,0,0,0]),row([0,0,0,0])]),TRUE_PARAMETER)
    assert abs(collision-(1-beta/e["inverse_temperature"]))<1e-12,collision
    return dict(rectangle_reference=reference,weighted_bridge_QMC=qmc_value,
                log_secondary_extreme_counts=powers,limiting_sib_count_exponent=expected,
                exact_pair_collision=collision)
