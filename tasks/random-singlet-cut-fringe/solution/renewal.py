"""Competing decimations of two marked cuts enclosing one or two active spins."""
import numpy as np
from scipy.integrate import solve_ivp
from apparatus import pair_response


# Both centers intact, generic one-center, altered left/right bonds, pending
# common bond, and the two renewal stages of a regenerated common cut.
TWO,ONE,LEFT,RIGHT,PENDING,SLOW,FAST=range(7)
lo=(3-np.sqrt(5))/2
hi=3-lo


def response(q,instant=False):
    count=q['count']
    levels=1 if count is None else count+1
    n=14*levels
    def run(use_spin):
        def idx(k,color,stage):return (2*k+color)*7+stage
        state=np.zeros(n,complex)
        for c,s in enumerate((-1,1)):
            state[idx(0,c,ONE if q['central']==1 else TWO)]=.5 if q['left_sign'] is None else float(q['left_sign']==s)
        now=0.
        for band in q['bands']:
            A=np.zeros((n,n),complex)
            middle=np.zeros_like(A)
            def event(k,c,stage,newstage,rate,matrix,spot=None,spin=None,flip=False):
                col=idx(k,c,stage)
                matrix[col,col]-=rate
                nc=1-c if flip else c
                channel=idx(k,nc,newstage)
                m,p=(1.,0.) if spot is None else optics[spot,spin]
                if not use_spin:m=1.
                if count is None:p=0.
                matrix[channel,col]+=rate*m*(1-p)
                if k+1<levels:matrix[channel+14,col]+=rate*m*p
            optics={(spot,s):pair_response(band[spot],s) for spot in ('left','right','both') for s in (-1,1)}
            for k in range(levels):
                for c,sign in enumerate((-1,1)):
                    event(k,c,TWO,SLOW,1,middle)
                    event(k,c,TWO,LEFT,1,A,'left',sign,True)
                    event(k,c,TWO,RIGHT,1,A,'right',sign)
                    event(k,c,LEFT,ONE,1,middle)
                    event(k,c,RIGHT,ONE,1,middle)
                    merged=SLOW if instant else PENDING
                    event(k,c,LEFT,merged,1,A,'right',-sign)
                    event(k,c,RIGHT,merged,1,A,'left',sign,True)
                    event(k,c,ONE,SLOW,1,A,'left',sign,True)
                    event(k,c,ONE,SLOW,1,A,'right',-sign)
                    event(k,c,PENDING,SLOW,1,middle)
                    event(k,c,SLOW,FAST,lo,A)
                    event(k,c,FAST,SLOW,hi,A,'both',sign,True)
            end=now+band['width']
            ans=solve_ivp(lambda t,y:A@y+np.exp(t)*(middle@y),(now,end),state,method='DOP853',
                          rtol=1.e-10,atol=2.e-13)
            state=ans.y[:,-1];now=end
        return state[-14:].sum()
    rate=run(False).real
    return rate,(np.exp(-1j*q['phase'])*run(True)).real/rate
