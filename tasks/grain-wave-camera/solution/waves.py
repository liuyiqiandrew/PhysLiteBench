from functools import lru_cache
from itertools import combinations,product
import numpy as np


def resistance(e,at=1.):
    _,_,length,c,d=e
    return length*(min(.5,at)/c+max(0.,at-.5)/d)


@lru_cache(maxsize=128)
def trees(n,tracks,src):
    R=np.array([resistance(e) for e in tracks])
    # Dimensionless macroscopic tree count; the common microscopic factors
    # cancel between the ordinary and the two-root tree partition functions.
    G=np.zeros((n,n))
    for (i,j,*_),r in zip(tracks,R):
        for p in (i,j):
            if p>=0:G[p,p]+=1/r
        if i>=0 and j>=0:G[i,j]-=1/r;G[j,i]-=1/r
    Z=np.linalg.det(G)
    out=[]
    for ids in combinations(range(len(tracks)),n-1):
        parent=list(range(n+1));parent[-1]=src
        def root(i):
            i=i if i>=0 else n
            while parent[i]!=i:i=parent[i]
            return i
        for e in ids:
            i,j,*_=tracks[e];a,b=root(i),root(j)
            if a==b:break
            parent[a]=b
        else:
            adj=[[] for _ in parent]
            for e in ids:
                i,j,l,*_=tracks[e];adj[i].append((j,l));adj[j].append((i,l))
            dist=[None]*(n+1);dist[src]=0.;todo=[src]
            while todo:
                i=todo.pop()
                for j,l in adj[i]:
                    if dist[j] is None:dist[j]=dist[i]+l;todo.append(j)
            missing=set(range(len(tracks)))-set(ids)
            exits={}
            for e in missing:
                i,j,l,c,d=tracks[e]
                if i==src and dist[j] is None:exits[e]=c
                if j==src and dist[i] is None:exits[e]=d
            out.append((dist,missing,exits,1/np.prod(R[list(ids)])/Z))
    return out


def clocks(q,tracks,dist,missing,src,pins,rate):
    groups={}
    for e,r in q["pickup"]:
        if e in missing:groups.setdefault(e,set()).add(r)
    opt=[]
    for e,probes in groups.items():
        if e in pins:
            pos=0. if tracks[e][0]==src else 1.
            opt.append([(e,pos,1.)])
        else:
            x=[0.,*sorted(probes),1.]
            R=resistance(tracks[e])
            opt.append([(e,(a+b)/2,(resistance(tracks[e],b)-resistance(tracks[e],a))/R)
                        for a,b in zip(x,x[1:])])
    den=num=0.
    for pattern in product(*opt):
        cut={e:x for e,x,p in pattern}
        wt=np.prod([p for e,x,p in pattern])
        times=[]
        for e,r in q["pickup"]:
            i,j,l,*_=tracks[e]
            avail=[]
            if dist[i] is not None and (e not in cut or r<cut[e]):avail.append(dist[i]+l*r)
            if dist[j] is not None and (e not in cut or r>cut[e]):avail.append(dist[j]+l*(1-r))
            times.append(min(avail) if avail else None)
        g,p=times
        if g is None or not q["rounds"][0]<=g<=q["rounds"][1]:continue
        den+=wt*np.exp(-rate*g)
        if p is not None and q["delay"][0]<=p-g<=q["delay"][1]:num+=wt*np.exp(-rate*max(g,p))
    return den,num


def masses(q,domain,rate):
    tracks=tuple(map(tuple,domain["tracks"]))
    src=q["drive"]
    ans=np.zeros((q['deficit']+1,2))
    for dist,missing,eligible,w in trees(q["sites"],tracks,src):
        for count in range(1,len(eligible)+1):
            for pins in combinations(eligible,count):
                if sum(eligible[e] for e in pins)!=q["deficit"]:continue
                p=np.prod([1/(eligible[e]*resistance(tracks[e])) for e in pins])
                den,num=clocks(q,tracks,dist,missing,src,pins,rate)
                ans[count]+=w*p*np.array([den,num])
    return ans


def signal(q, rate, reprepare=False):
    accepted = []
    for domain in q["domains"]:
        bins = masses(q, domain, rate)
        nonzero = np.flatnonzero(bins[:, 0])
        if len(nonzero):
            order = nonzero[0]
            den, num = bins[order]
            accepted.append((order, domain["weight"], den, num))
    if reprepare:
        scale = max(weight for _, weight, _, _ in accepted)
        return (sum((weight / scale) * (num / den) for _, weight, den, num in accepted)
                / sum(weight / scale for _, weight, _, _ in accepted))
    order = min(row[0] for row in accepted)
    leading = [row for row in accepted if row[0] == order]
    scale = max(weight for _, weight, _, _ in leading)
    denominator = sum((weight / scale) * den for _, weight, den, _ in leading)
    numerator = sum((weight / scale) * num for _, weight, _, num in leading)
    return numerator / denominator
