import numpy as np
from source import Sharp

class Readout:
    def fit(self,records):
        f=np.cos([r['input']['phase'] for r in records]);v=[r['value'] for r in records]
        self.visibility=float(np.clip(f@v/(f@f),.65,.95));return self
    def predict(self,inputs):
        return self.visibility*np.array([np.cos(q['phase']) if q.get('calibration') else self.fringe(q) for q in inputs],float)

def mean_reference(q):
    """Replace the dither on every exposure by its unconditional mean and use the slit midpoint."""
    c=q['center'];mid=sum(q['slit'])/2;p,k=q['lock'];lo,hi=q['dither'];s=Sharp(q)
    ref=p*k+(1-p)*(lo+hi)/2;d=mid-ref
    bound=.999999*min(c,s.edges[-1]-c)
    d=np.clip(d,-bound,bound)
    prob,coh=s.raw(c+d,c-d,diagonal=abs(d)<1e-13)
    if prob.real<1e-14:return 0.
    return (np.exp(-1j*(q['phase']+q['lag']*ref))*coh).real/prob.real
