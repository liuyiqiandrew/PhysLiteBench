"""Propagate atomic and continuous incident frequency measures through a differential dither."""
import numpy as np
from scipy.special import roots_legendre
from source import Sharp


def observe(q,independent_phase=False,drop_shared=False,degree=12):
    c=q['center'];a,b=q['slit'];p,k=q['lock'];lo,hi=q['dither'];omega=q['lag']
    source=Sharp(q);T=source.edges[-1];extent=min(c,T-c);density=(1-p)/(hi-lo)
    if q['kind']=='beat' and a>0:windows=[(a,b),(-b,-a)]
    elif q['kind']=='beat':windows=[(-b,b)]
    else:windows=[(a,b)]
    point=a==b;common=source.raw(c,c,True)
    def phase(x):return np.exp(-1j*omega*x)
    def average(x,y):
        return phase((x+y)/2)*np.sinc(omega*(y-x)/(2*np.pi))
    decorrelation=p*phase(k)+(1-p)*average(lo,hi)
    def moments(w,z):
        return np.array([w,w*decorrelation if independent_phase else z])
    locked=any(l==k for l,r in windows) if point else False
    if locked and p*common[0].real>1e-15:
        return common*moments(p,p*phase(k))
    def spread(d):
        mass=0.;coh=0j
        for l,r in windows:
            if point:
                x=l-d
                weight=density if lo<x<hi else density/2 if x in (lo,hi) else 0.
                mass+=weight;coh+=weight*phase(x)
            else:
                x,y=max(lo,l-d),min(hi,r-d)
                if x<y:mass+=density*(y-x);coh+=density*(y-x)*average(x,y)
        return moments(mass,coh)
    def pinned(d):
        if not point and any(l<=d+k<=r for l,r in windows):
            return moments(p,p*phase(k))
        return np.zeros(2,complex)
    answer=common*(pinned(0)+(0. if drop_shared else spread(0)))
    if point and p:
        for l,r in windows:
            d=l-k
            if abs(d)<=extent:
                borders=int(c+d in (0.,T))+int(c-d in (0.,T))
                boundary_weight=.5**borders
                if d==0:
                    smooth=.5*(source.raw(c,c,order='LR')+source.raw(c,c,order='RL'))
                else:smooth=source.raw(c+d,c-d)
                answer+=2*boundary_weight*smooth*moments(p,p*phase(k))
    nodes,weights=roots_legendre(degree)
    split={-extent,extent,0.}
    for edge in source.edges[1:-1]:split.update([edge-c,c-edge])
    for l,r in windows:
        for h in (l,r):
            split.update(h-x for x in (lo,hi,k))
    limits=sorted(d for d in split if -extent<=d<=extent)
    for l,r in zip(limits,limits[1:]):
        mid=(l+r)/2
        if (spread(mid)[0]+pinned(mid)[0]).real==0:continue
        for z,w in zip(nodes,weights):
            d=mid+(r-l)*z/2
            answer+=(r-l)*w*source.raw(c+d,c-d)*(spread(d)+pinned(d))
    return answer

def reading(q,**options):
    if q.get('calibration'):return 1.,np.cos(q['phase'])
    p,z=observe(q,**options);return p.real,(np.exp(-1j*q['phase'])*z).real/p.real
