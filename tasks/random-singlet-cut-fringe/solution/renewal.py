"""Alternating crossing orientations with the two-stage waiting law of a marked bond."""
import numpy as np
from scipy.linalg import expm
from apparatus import pair_response


def response(q,independent_herald=False):
    slow=(3-np.sqrt(5))/2
    fast=3-slow
    def average(signal,count):
        levels=1 if count is None else count+1
        state=np.zeros((levels,2,2),complex)
        orientations=[.5,.5] if q['last'] is None else [q['last']==1,q['last']==-1]
        phase=[1.,0.] if q['start']=='crossing' else [fast/3,slow/3]
        state[0]=np.outer(orientations,phase)
        state=state.ravel()
        for band in q['bands']:
            matrix=np.zeros((len(state),len(state)),complex)
            for o,orientation in enumerate((-1,1)):
                overlap,click=pair_response(band,orientation)
                gain=overlap if signal else 1.
                if count is None:click=0.
                for k in range(levels):
                    a=4*k+2*o;b=a+1
                    matrix[a,a]=-slow
                    matrix[b,a]=slow
                    matrix[b,b]=-fast
                    target=4*k+2*(1-o)
                    matrix[target,b]+=fast*gain*(1-click)
                    if k+1<levels:matrix[target+4,b]+=fast*gain*click
            state=expm(band['width']*matrix)@state
        return state[-4:].sum()
    acceptance=average(False,q['count']).real
    if independent_herald:
        coherence=acceptance*average(True,None)
    else:
        coherence=average(True,q['count'])
    return acceptance,(np.exp(-1j*q['phase'])*coherence).real/acceptance
