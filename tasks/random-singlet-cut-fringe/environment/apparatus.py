"""Uniform rotations, staggered rotations, and a singlet analyzer on an individual pair."""
import numpy as np
from scipy.linalg import expm

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
    """Constant-rate independent crossings, with the same disorder shared by the two copies."""
    count=q['count']
    levels=1 if count is None else count+1
    first=q['last']
    initial=np.array([.5,.5] if first is None else [first==1,first==-1],float)
    def run(signal):
        mass=np.zeros(2*levels,complex); mass[:2]=initial
        for band in q['bands']:
            generator=-np.eye(len(mass),dtype=complex)/3
            for k in range(levels):
                for s,sign in enumerate((-1,1)):
                    gain,p=pair_response(band,sign)
                    if not signal:gain=1.
                    if count is None:p=0.
                    col=2*k+s
                    generator[2*k+1-s,col]+=gain*(1-p)/3
                    if k+1<levels:generator[2*(k+1)+1-s,col]+=gain*p/3
            mass=expm(band['width']*generator)@mass
        return mass[-2:].sum()
    return float((np.exp(-1j*q['phase'])*run(True)).real/run(False).real)
