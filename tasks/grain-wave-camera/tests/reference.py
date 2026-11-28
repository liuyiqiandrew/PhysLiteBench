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


@lru_cache(maxsize=32)
def parents(links,n,s):
    from itertools import product
    choices=[[(None,None)] if j==s else [] for j in range(n)]
    for e,(a,b,l) in enumerate(links):
        if a>=0 and a!=s:choices[a].append((b,e))
        if b>=0 and b!=s:choices[b].append((a,e))
    arr=[]
    for assignment in product(*choices):
        depth=[]
        for x in range(n):
            visited=set();d=0;z=x
            while z not in (s,-1) and z not in visited:
                visited.add(z);z,e=assignment[z];d+=links[e][2]
            if z in visited:break
            depth.append(d if z==s else None)
        else:
            kept=set(e for z,e in assignment if e is not None)
            w=1./np.prod([links[e][2] for e in kept])
            arr.append((depth+[None],kept,w))
    return arr


def finite(q,rate,N):
    from itertools import product
    links=tuple(map(tuple,q["tracks"]));src=q["drive"]
    num=den=0.
    for depth,kept,wt in parents(links,q["sites"],src):
        # The coarse forest is oriented towards its two roots. Place the
        # omitted bond on each other track of N*l physical steps.
        cuts={}
        for i,(a,b,l) in enumerate(links):
            if i in kept:continue
            ns=N*l
            bounds={0,ns}
            for e,r in q["pickup"]:
                if i==e:bounds.add(round(r*ns))
            if src==a:bounds.add(1)
            if src==b:bounds.add(ns-1)
            h=sorted(bounds)
            cuts[i]=[(y,(y-x)/ns) for x,y in zip(h,h[1:])]
        for pattern in product(*cuts.values()):
            cut={e:j for e,(j,p) in zip(cuts,pattern)}
            w=wt*np.prod([p for j,p in pattern])
            closed=[]
            for e,j in cut.items():
                a,b,l=links[e]
                if (src==a and j==1 and depth[b] is None or
                    src==b and j==N*l and depth[a] is None):
                    closed.append(e)
            if q["class"]=="last" and not closed:continue
            if q["class"]=="port" and not set(closed).intersection(q["ports"]):continue
            ds=[]
            for e,r in q["pickup"]:
                a,b,l=links[e];m=round(N*l*r)
                candidates=[]
                if depth[a] is not None and (e in kept or cut[e]>m):
                    candidates.append(depth[a]+m/N)
                if depth[b] is not None and (e in kept or cut[e]<=m):
                    candidates.append(depth[b]+l-m/N)
                ds.append(min(candidates) if candidates else None)
            g,p=ds
            if g is None or not q["rounds"][0]<=g<=q["rounds"][1]:continue
            den+=w*np.exp(-rate*g)
            if p is not None and q["delay"][0]<=p-g<=q["delay"][1]:
                num+=w*np.exp(-rate*max(g,p))
    return num/den
