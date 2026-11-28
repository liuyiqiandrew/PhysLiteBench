"""Survival evolution between resolved spectral lines and a marked jump at each retained line."""
import numpy as np
from scipy.integrate import solve_ivp
from apparatus import optics,mask

low=(3-np.sqrt(5))/2;high=3-low
def measurement(q,independent=False):
    records=q['lines'];marks={t for t in records if t is not None}
    edges=np.r_[0.,np.cumsum([b['width'] for b in q['bands']])]
    def run(signal):
        state=np.zeros((2,7),complex)
        for c,s in enumerate((-1,1)):
            first={0:5,1:1,2:0}[q['central']]
            state[c,first]=.5 if q['left_sign'] is None else float(q['left_sign']==s)
        state=state.ravel()
        for j,b in enumerate(q['bands']):
            opt={(g,s):optics(b[g],g,s,independent) for g in ('left','right','both') for s in (-1,1)}
            def evolution(t,hit):
                fixed=np.zeros((14,14),complex);inner=np.zeros_like(fixed)
                allowed=mask(records,t,hit)
                def move(old,new,r,mat,geom=None,sign=None,flip=False):
                    source=7*c+old;target=7*(1-c if flip else c)+new
                    if not hit:mat[source,source]-=r
                    if geom is None:
                        if not hit:mat[target,source]+=r
                    else:
                        m,p=opt[geom,sign]
                        mat[target,source]+=r*(p@allowed)*(m if signal else 1.)
                for c,s in enumerate((-1,1)):
                    move(0,5,1,inner)
                    move(0,2,1,fixed,'left',s,True);move(0,3,1,fixed,'right',s)
                    move(2,1,1,inner);move(3,1,1,inner)
                    move(2,4,1,fixed,'right',-s);move(3,4,1,fixed,'left',s,True)
                    move(1,5,1,fixed,'left',s,True);move(1,5,1,fixed,'right',-s)
                    move(4,5,1,inner);move(5,6,low,fixed);move(6,5,high,fixed,'both',s,True)
                return fixed,inner
            l,r=edges[j:j+2];cuts=[l,*sorted(v for v in marks if l<v<r),r]
            for a,bound in zip(cuts,cuts[1:]):
                A,B=evolution(.5*(a+bound),False)
                sol=solve_ivp(lambda u,y:A@y+np.exp(u)*(B@y),(a,bound),state,method='DOP853',rtol=4e-11,atol=2e-13)
                state=sol.y[:,-1]
                if bound in marks:
                    jump,extra=evolution(bound,True)
                    state=(jump+np.exp(bound)*extra)@state
        return state.sum()
    density=run(False).real
    return density,(np.exp(-1j*q['phase'])*run(True)).real/density
