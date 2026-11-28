"""Condition Gaussian Stark shifts at fixed telegraph histories and Poisson occupancies."""
import itertools
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre,gammaln,xlogy


@lru_cache(maxsize=None)
def compositions(n,k):
    if k==1:
        return [(n,)]
    return [(i,)+v for i in range(n+1) for v in compositions(n-i,k-1)]


def kernel(pop, direction, kicks, x, ceiling):
    count,sigma,gamma=pop
    x=np.atleast_1d(x)
    coefficients=[];prob=[]
    for states in itertools.product((1,-1),repeat=len(kicks)):
        weight=1.;phase=0.;last=0.;old=1
        for s,(t,a,b) in zip(states,kicks):
            weight*=(1+s*old*np.exp(-2*gamma*(t-last)))/2
            phase+=s*(a*direction[0]+b*direction[1])
            last=t;old=s
        coefficients.append(phase);prob.append(weight)
    h=np.asarray(coefficients);prob=np.asarray(prob)
    ans=np.zeros((len(x),2),complex)
    for n in range(1,ceiling+1):
        hist=np.array(compositions(n,len(h)))
        weights=np.exp(-count+n*np.log(count)+xlogy(hist,prob).sum(1)-gammaln(hist+1).sum(1))
        mean=hist@h/n
        spread=hist@(h*h)-n*mean**2
        gaussian=np.exp(-.5*x*x/(sigma*sigma*n))/np.sqrt(2*np.pi*n*sigma*sigma)
        ans[:,0]+=gaussian*weights.sum()
        ans[:,1]+=gaussian*(np.exp(-1j*x[:,None]*mean)@(weights*np.exp(-.5*sigma*sigma*spread)))
    return ans


def truth(q,ceiling=16,n_quad=125):
    if q.get('blank'):return np.cos(q['phase'])
    a,b=q['gate'];u,v=q['mix']
    meter=np.array([[1.,u],[v,1.]])
    readings=np.unique([[a,b],[a,-b]] if q['unsigned_b'] else [[a,b]],axis=0)
    choices=[]
    for domain in q['domains']:
        r=domain['ratio'];directions=np.array([[1.,0.],[0.,1.],[1.,r]])
        columns=directions@meter.T
        for target in readings:
            for mask in itertools.product((False,True),repeat=3):
                used=np.flatnonzero(mask)
                matrix=columns[used].T
                if not len(used):
                    best=np.empty(0);rank=0
                else:
                    best,_,rank,_=np.linalg.lstsq(matrix,target,rcond=None)
                if np.max(abs(matrix@best-target))<1e-10:
                    choices.append((rank,domain,directions,columns,used,best))
    rank=min(item[0] for item in choices)
    results=np.zeros(2,complex)
    for degree,dom,directions,columns,used,best in choices:
        if degree!=rank:continue
        pockets=dom['pockets'];w=dom['weight']*np.exp(-sum(p[0] for i,p in enumerate(pockets) if i not in used))
        def density(i,x):
            return kernel(pockets[i],directions[i],q['kicks'],x,ceiling)
        if not degree:
            results+=w
        elif degree==1:
            j=used[0];results+=w*density(j,best[0])[0]/np.max(abs(columns[j]))
        elif len(used)==2:
            i,j=used
            results+=w*(density(i,best[0])*density(j,best[1]))[0]/abs(np.linalg.det(columns[used]))
        else:
            # The null vector parameterizes the exact affine intersection in the three Stark shifts.
            _,_,v_t=np.linalg.svd(columns.T,full_matrices=True)
            normal=v_t[-1];z,weights=roots_legendre(n_quad)
            width=18.;coords=best+width*z[:,None]*normal
            area=width/np.sqrt(np.linalg.det(columns.T@columns))
            value=np.prod([density(i,coords[:,i]) for i in used],axis=0)
            results+=w*area*(weights@value)
    return float((np.exp(-1j*q['phase'])*results[1]).real/results[0].real)
