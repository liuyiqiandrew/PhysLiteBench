from functools import lru_cache
from itertools import combinations,product
import numpy as np


@lru_cache(maxsize=48)
def forests(n,links,src):
    roots=(src,n)
    edges=[(i if i>=0 else n,j if j>=0 else n,l) for i,j,l in links]
    ans=[]
    for used in combinations(range(len(edges)),n-1):
        parent=list(range(n+1));parent[n]=src
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]];i=parent[i]
            return i
        for e in used:
            i,j,_=edges[e];x,y=find(i),find(j)
            if x==y:break
            parent[x]=y
        else:
            adj=[[] for _ in parent]
            for e in used:
                i,j,l=edges[e];adj[i].append((j,l));adj[j].append((i,l))
            # Distances (in macroscopic parallel rounds) exist only in the source component.
            dep=[None]*(n+1);dep[src]=0.;todo=[src]
            while todo:
                i=todo.pop()
                for j,l in adj[i]:
                    if dep[j] is None:dep[j]=dep[i]+l;todo.append(j)
            missing=set(range(len(edges)))-set(used)
            exits=[e for e in missing if src in edges[e][:2] and
                   dep[edges[e][1] if edges[e][0]==src else edges[e][0]] is None]
            wt=np.prod([edges[e][2] for e in missing],dtype=float)
            ans.append((edges,dep,missing,exits,wt))
    return ans


def signal(q,rate,coarse=False):
    links=tuple(map(tuple,q["tracks"]))
    num=den=0.
    for edges,dep,missing,exits,wt in forests(q["sites"],links,q["drive"]):
        if q["class"]=="all":
            pins=[(None,1.)]
        else:
            pins=[(e,1. if coarse else 1./edges[e][2]) for e in exits if q["class"]=="last" or e in q["ports"]]
        for pin,pinweight in pins:
            cut={}
            for e,r in q["pickup"]:
                if e in missing:
                    key=e
                    cut.setdefault(key,set()).add(r)
            choices=[]
            for key,probes in cut.items():
                e=key
                if e==pin:
                    pos=0. if edges[e][0]==q["drive"] else 1.
                    choices.append([(key,pos,1.)])
                else:
                    v=[0.,*sorted(probes),1.]
                    choices.append([(key,(a+b)/2,b-a) for a,b in zip(v,v[1:])])
            for pattern in product(*choices):
                pos={k:x for k,x,_ in pattern}
                w=wt*pinweight*np.prod([p for _,_,p in pattern])
                d=[]
                for e,r in q["pickup"]:
                    i,j,l=edges[e]
                    if e not in missing:
                        if dep[i] is None:t=None
                        else:t=min(dep[i]+r*l,dep[j]+(1-r)*l)
                    elif r<pos[e]:
                        t=None if dep[i] is None else dep[i]+r*l
                    else:
                        t=None if dep[j] is None else dep[j]+(1-r)*l
                    d.append(t)
                g,p=d
                if g is None or not q["rounds"][0]<=g<=q["rounds"][1]:continue
                den+=w*np.exp(-rate*g)
                if p is not None and q["delay"][0]<=p-g<=q["delay"][1]:
                    num+=w*np.exp(-rate*max(g,p))
    return num/den
