"""Uniform rotations, staggered rotations, and a singlet analyzer on an individual pair."""
import numpy as np

pauli={
    'x':np.array([[0.,1.],[1.,0.]],complex),
    'y':np.array([[0.,-1j],[1j,0.]]),
    'z':np.diag([1.,-1.]),
}
singlet=np.array([0.,1.,-1.,0.])/np.sqrt(2.)


def pair_response(band,left_sign):
    """Return complex Ramsey overlap and click probability for one crossing."""
    states=[]
    for pulses in band['arms']:
        state=singlet
        for side,axis,uniform,staggered in pulses:
            spin_sign=left_sign if side=='L' else -left_sign
            a=(uniform+spin_sign*staggered)/2
            u=np.cos(a)*np.eye(2)-1j*np.sin(a)*pauli[axis]
            gate=np.kron(u,np.eye(2)) if side=='L' else np.kron(np.eye(2),u)
            state=gate@state
        states.append(state)
    efficiency,uniform,staggered=band['herald']
    click=efficiency*np.sin(.5*(uniform+left_sign*staggered))**2
    return np.vdot(*states),click


class Readout:
    def fit(self,records):
        basis=np.cos([r['input']['phase'] for r in records])
        values=np.array([r['value'] for r in records])
        self.visibility=float(np.clip(basis@values/(basis@basis),.65,.95))
        return self

    def predict(self,inputs):
        return self.visibility*np.array([self.fringe(q) for q in inputs],float)


def poisson(q):
    """Three independent stationary streams, each with rate 1/6 per log-scale unit."""
    count=q['count']
    def average(signal):
        offset=0j; tagged=0j
        for band in q['bands']:
            for spot in ('left','right','both'):
                for sign in (-1,1):
                    gain,click=pair_response(band[spot],sign)
                    if not signal:gain=1.
                    if count is None:click=0.
                    lam=band['width']/12
                    offset+=lam*(gain*(1-click)-1)
                    tagged+=lam*gain*click
        return np.exp(offset)*(1 if count is None else tagged**count)
    rate=average(False).real
    return 0. if rate==0 else float((np.exp(-1j*q['phase'])*average(True)).real/rate)
