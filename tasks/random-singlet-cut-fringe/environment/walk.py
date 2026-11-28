"""Continuum probabilities of two-cut singlets for arbitrary linear measurement records."""
from itertools import product
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bsr_matrix

slow=(3-np.sqrt(5))/2;fast=3-slow
spots=('left','right','both')

def advance(q,start,step):
    """Return terminal record vectors and region status (separate or merged), after herald selection.

    start is the initial record vector. step(band,spot,s,merges) returns an
    update and a herald click probability for an event. spot=None is a
    merger with no detected singlet. All updates apply in formation order.
    """
    requested=q['counts']
    dims=tuple(1 if c is None else c+1 for c in requested)
    indices=list(product(*(range(k) for k in dims)));lookup={key:i for i,key in enumerate(indices)}
    d=len(start);K=len(indices);n=14*K;eye=np.eye(d)
    def at(c,o,i):return 14*c+7*o+i
    state=np.zeros((K,2,7,d),complex)
    for o,s in enumerate((-1,1)):
        mass=.5 if q['left_sign'] is None else float(q['left_sign']==s)
        state[0,o,int(q['central']==1)]=mass*start
    state=state.ravel()
    def sparse(table):
        rows=[[] for _ in range(n)]
        for (i,j),m in table.items():rows[i].append((j,m))
        ptr=[0];col=[];val=[]
        for row in rows:
            for j,m in sorted(row):col.append(j);val.append(m)
            ptr.append(len(col))
        return bsr_matrix((np.asarray(val),col,ptr),shape=(n*d,n*d))
    def add(table,row,col,m):
        if (row,col) not in table:table[row,col]=m.copy()
        else:table[row,col]+=m
    t=0.
    for band in q['bands']:
        const={};rate={}
        ops={(spot,s,merges):step(band,spot,s,merges) for spot in spots for s in (-1,1) for merges in (False,True)}
        fusion=step(band,None,1,True)[0]
        def move(a,b,r,table,spot=None,sign=None,flip=False,merges=False):
            src=at(c,o,a);dest=1-o if flip else o
            add(table,src,src,-r*eye)
            if spot is None:
                add(table,at(c,dest,b),src,r*(fusion if merges else eye));return
            op,p=ops[spot,sign,merges];j=spots.index(spot)
            if requested[j] is None:p=0.
            add(table,at(c,dest,b),src,r*(1-p)*op)
            if key[j]+1<dims[j]:
                k=list(key);k[j]+=1
                add(table,at(lookup[tuple(k)],dest,b),src,r*p*op)
        for c,key in enumerate(indices):
            for o,s in enumerate((-1,1)):
                move(0,5,1,rate,merges=True)
                move(0,2,1,const,'left',s,True);move(0,3,1,const,'right',s)
                move(2,1,1,rate);move(3,1,1,rate)
                move(2,4,1,const,'right',-s,merges=True)
                move(3,4,1,const,'left',s,True,True)
                move(1,5,1,const,'left',s,True,True);move(1,5,1,const,'right',-s,merges=True)
                move(4,5,1,rate)
                move(5,6,slow,const);move(6,5,fast,const,'both',s,True)
        A=sparse(const);B=sparse(rate);end=t+band['width']
        sol=solve_ivp(lambda u,y:A@y+np.exp(u)*(B@y),(t,end),state,method='DOP853',rtol=3e-9,atol=1e-11)
        t=end;state=sol.y[:,-1]
    final=state.reshape(K,2,7,d)[-1].sum(axis=0)
    return final[:4].sum(axis=0),final[4:].sum(axis=0)
