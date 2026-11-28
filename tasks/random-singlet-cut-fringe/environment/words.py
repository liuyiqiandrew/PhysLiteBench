"""Ordered matrix moments of the two-cut exchange spectrum."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bsr_matrix

slow=(3-np.sqrt(5))/2
fast=3-slow


def ordered_average(q,event,terminal):
    """Average M_first ... M_last times terminal, including the requested herald indicator.

    event(band, geometry, sign) returns the geometry's square matrix M and
    probability of a click in the herald copy for this singlet's left sign.
    """
    count=q['count'];K=1 if count is None else count+1
    terminal=np.asarray(terminal,complex);d=len(terminal);n=14*K
    I=np.eye(d)
    def index(k,o,s):return 7*(2*k+o)+s
    def sparse(table):
        rows=[[] for _ in range(n)]
        for (i,j),a in table.items():rows[i].append((j,a))
        indices=[];data=[];indptr=[0]
        for row in rows:
            for j,a in sorted(row):indices.append(j);data.append(a)
            indptr.append(len(data))
        return bsr_matrix((np.array(data),indices,indptr),shape=(n*d,n*d))
    def put(table,i,j,a):
        if (i,j) not in table:table[i,j]=a.copy()
        else:table[i,j]+=a
    value=np.zeros((K,2,7,d),complex);value[0]=terminal
    value=value.ravel()
    ends=np.r_[0.,np.cumsum([b['width'] for b in q['bands']])]
    for j in reversed(range(len(q['bands']))):
        band=q['bands'][j]
        fixed={};inside={}
        marks={(spot,s):event(band,spot,s) for spot in ('left','right','both') for s in (-1,1)}
        def jump(old,new,rate,table,spot=None,sign=None,flip=False):
            src=index(k,o,old)
            put(table,src,src,-rate*I)
            m,p=(I,0.) if spot is None else marks[spot,sign]
            if count is None:p=0.
            other=1-o if flip else o
            put(table,src,index(k,other,new),rate*(1-p)*m)
            if k:put(table,src,index(k-1,other,new),rate*p*m)
        for k in range(K):
            for o,s in enumerate((-1,1)):
                jump(0,5,1,inside)
                jump(0,2,1,fixed,'left',s,True)
                jump(0,3,1,fixed,'right',s)
                jump(2,1,1,inside);jump(3,1,1,inside)
                jump(2,4,1,fixed,'right',-s);jump(3,4,1,fixed,'left',s,True)
                jump(1,5,1,fixed,'left',s,True);jump(1,5,1,fixed,'right',-s)
                jump(4,5,1,inside)
                jump(5,6,slow,fixed)
                jump(6,5,fast,fixed,'both',s,True)
        a=sparse(fixed);b=sparse(inside)
        result=solve_ivp(lambda t,y:-(a@y+np.exp(t)*(b@y)),(ends[j+1],ends[j]),value,
                         rtol=5e-9,atol=7e-12,method='DOP853')
        value=result.y[:,-1]
    start=value.reshape(K,2,7,d)[-1,:,int(q['central']==1)]
    return start.mean(axis=0) if q['left_sign'] is None else start[int(q['left_sign']==1)]
