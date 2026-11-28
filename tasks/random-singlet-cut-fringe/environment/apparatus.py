"""Carrier preparation, turnaround, readout, and a reference instrument with independent partner spins."""
import numpy as np
from words import ordered_average

pauli={'x':np.array([[0.,1.],[1.,0.]],complex),'y':np.array([[0.,-1j],[1j,0.]]),'z':np.diag([1.,-1.])}


def pure(raw):
    x=np.array([complex(*w) for w in raw]);x/=np.linalg.norm(x)
    return np.outer(x,x.conj())


def exchange(a,b):
    """One encounter with an independent unpolarized spin. The bra pulse is a, the ket pulse b."""
    factor=np.cos(a)*np.cos(b)+.5j*np.sin(a-b)
    return factor*np.eye(4)+.5*np.sin(a)*np.sin(b)*np.outer(np.eye(2).ravel(),np.eye(2).ravel())


def independent(early,late):
    return np.kron(exchange(*late),exchange(*early).T)


def interferometer(q,partner=independent):
    initial=pure(q['prepare'])
    analyzer=np.eye(2) if q['analyze'] is None else pure(q['analyze'])
    mids=[np.cos(t/2)*np.eye(2)-1j*np.sin(t/2)*pauli[axis] for axis,t in q['turnaround']]
    def channel(a,b):
        def event(band,geometry,s):
            spec=band[geometry]
            angles=np.asarray(spec['arms'])
            early=angles[:,0]+s*angles[:,1];late=angles[:,2]-s*angles[:,3]
            eta,h,g=spec['herald']
            return partner(early[[a,b]],late[[a,b]]),eta*np.sin(.5*(h+s*g))**2
        terminal=np.kron(mids[b],mids[a].conj()).ravel()
        word=ordered_average(q,event,terminal).reshape(4,4)
        output=(word@initial.ravel()).reshape(2,2)
        return np.trace(analyzer@output)
    if q['analyze'] is None:
        def counts(band,geometry,s):
            eta,h,g=band[geometry]['herald']
            return np.ones((1,1)),eta*np.sin(.5*(h+s*g))**2
        p=ordered_average(q,counts,[1.])[0].real
    else:p=.5*(channel(0,0)+channel(1,1)).real
    cross=channel(0,1)
    return p,float((np.exp(-1j*q['phase'])*cross).real/p)


class Readout:
    def fit(self,records):
        phase=np.cos([r['input']['phase'] for r in records]);y=[r['value'] for r in records]
        self.visibility=float(np.clip(phase@y/(phase@phase),.65,.95))
        return self
    def predict(self,inputs):
        return self.visibility*np.array([self.fringe(q) for q in inputs],float)
