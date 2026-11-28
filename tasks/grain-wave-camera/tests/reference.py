"""Enumerate recurrent heights, add a grain, and record every parallel wave."""
from collections import Counter
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=32)
def literal(n,links,src):
    A=np.zeros((n,n),int);degree=np.zeros(n,int)
    for i,j in links:
        if i>=0:degree[i]+=1
        if j>=0:degree[j]+=1
        if i>=0 and j>=0:A[i,j]+=1;A[j,i]+=1
    hist=Counter(); seen=0
    # Stable configurations use mixed-radix site heights. Burn from the
    # drain to classify recurrence, then perform literal grain transfers.
    for start in range(0,int(degree.prod()),18000):
        code=np.arange(start,min(start+18000,degree.prod()))
        h=np.empty((len(code),n),int)
        for j,d in enumerate(degree):h[:,j]=code%d;code=code//d
        alive=np.ones_like(h)
        for _ in range(n):alive=alive*(h<alive@A)
        h=h[np.sum(alive,1)==0];seen+=len(h)
        h[:,src]+=1
        while True:
            h=h[h[:,src]>=degree[src]]
            if len(h)==0:break
            active=np.zeros_like(h);active[:,src]=1;dist=np.full_like(h,-1)
            step=0
            while active.any():
                dist[active.astype(bool)]=step
                h+=active@A-active*degree
                active=(h>=degree).astype(int);active[:,src]=0
                step+=1
            final=h[:,src]<degree[src]
            merged=np.column_stack([dist,final.astype(int)])
            counts,nums=np.unique(merged,axis=0,return_counts=True)
            for row,num in zip(counts,nums):hist[tuple(row[:-1]),bool(row[-1])]+=int(num)
    return hist,seen


@lru_cache(maxsize=48)
def oriented(tracks,n,src):
    from itertools import product
    branches=[[] for j in range(n)]
    for e,(i,j,*tail) in enumerate(tracks):
        if i>=0:branches[i].append((j,e))
        if j>=0:branches[j].append((i,e))
    R=[l*(1/c+1/d)/2 for i,j,l,c,d in tracks]
    def collect(roots):
        allowed=[[(None,None)] if i in roots else ls for i,ls in enumerate(branches)]
        for perm in product(*allowed):
            dist=[]
            for i in range(n):
                history=set();k=i;t=0.
                while k not in roots and k not in history:
                    history.add(k);k,e=perm[k];t+=tracks[e][2]
                if k in history:break
                dist.append(t if k==src else None)
            else:
                feats={e for k,e in perm if k is not None}
                yield dist+[None],feats,1./np.prod([R[e] for e in feats])
    Z=sum(w for ds,kept,w in collect({-1}))
    return [(ds,kept,w/Z) for ds,kept,w in collect({-1,src})]


def finite_mass(q,d,N,rate):
    from itertools import product
    tracks=tuple(map(tuple,d['tracks']));src=q['drive']
    num=den=0.
    for dep,kept,weight in oriented(tracks,q["sites"],src):
        cats={}
        for e,(i,j,l,c,d2) in enumerate(tracks):
            if e in kept:continue
            length=l*N
            bounds={0,length//2,length}
            if i==src:bounds.add(1)
            if j==src:bounds.add(length-1)
            for k,r in q["pickup"]:
                if k==e:bounds.add(round(r*length))
            b=sorted(bounds)
            cats[e]=[(y,(y-x)/(c if y<=length//2 else d2)/(length*(1/c+1/d2)/2))
                     for x,y in zip(b,b[1:])]
        for seq in product(*cats.values()):
            cuts={e:y for e,(y,p) in zip(cats,seq)}
            returned=0
            for e,jcut in cuts.items():
                a,b,l,c,d2=tracks[e]
                if a==src and dep[b] is None and jcut==1:returned+=c
                if b==src and dep[a] is None and jcut==l*N:returned+=d2
            if returned!=q['deficit']:continue
            ds=[]
            for e,r in q["pickup"]:
                a,b,l,c,d2=tracks[e];m=round(N*l*r);dist=[]
                if dep[a] is not None and (e in kept or cuts[e]>m):dist.append(dep[a]+m/N)
                if dep[b] is not None and (e in kept or cuts[e]<=m):dist.append(dep[b]+l-m/N)
                ds.append(min(dist) if dist else None)
            g,p=ds
            if g is None or not q['rounds'][0]<=g<=q['rounds'][1]:continue
            weight_m=weight*np.prod([a for _,a in seq])
            den+=weight_m*np.exp(-rate*g)
            if p is not None and q['delay'][0]<=p-g<=q['delay'][1]:num+=weight_m*np.exp(-rate*max(p,g))
    return np.array([den,num])


def finite_joint(q,N,rate):
    den,num=sum(d["weight"]*finite_mass(q,d,N,rate) for d in q["domains"])
    return num/den
