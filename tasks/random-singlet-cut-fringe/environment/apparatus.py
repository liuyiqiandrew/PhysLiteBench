"""Exchange and magnetic precession of a carrier, projected on a single singlet."""
import numpy as np
from walk import advance

pauli={'x':np.array([[0.,1.],[1.,0.]],complex),'y':np.array([[0.,-1j],[1j,0.]]),'z':np.diag([1.,-1.])}
swap=np.eye(4).reshape(2,2,2,2).transpose(1,0,2,3).reshape(4,4)
s=np.array([0.,1.,-1.,0.])/np.sqrt(2);singlet=np.outer(s,s).reshape(2,2,2,2)

def rotate(field):
    d,t=field
    return np.cos(t/2)*np.eye(2)-1j*np.sin(t/2)*pauli[d]

def pair_map(spec,sign,bra,ket):
    """Map a flattened four-by-four interior carrier superoperator through one outer arc."""
    def gate(t,f):
        exchange=np.cos(t)*np.eye(4)-1j*np.sin(t)*swap
        return (np.kron(rotate(f),np.eye(2))@exchange).reshape(2,2,2,2)
    arm=np.asarray(spec['arms']);e=arm[:,0]+sign*arm[:,1];l=arm[:,2]-sign*arm[:,3]
    early,late=spec['field']
    M=np.einsum('iwls,IwLt,kzjx,KzJy,xsyt->iIjJlLkK',gate(l[ket],late),gate(l[bra],late).conj(),
                gate(e[ket],early),gate(e[bra],early).conj(),singlet,optimize=True)
    efficiency,h,g=spec['herald']
    return M.reshape(16,16),efficiency*np.sin((h+sign*g)/2)**2

def pure(raw):
    amp=np.array([complex(*n) for n in raw]);return np.outer(amp,amp.conj())/np.vdot(amp,amp)

def fringe(q,records):
    rho=pure(q['prepare']);F=np.eye(2) if q['analyze'] is None else pure(q['analyze'])
    U=[rotate(spec) for spec in q['launch']]
    def signal(a,b):
        layers=records(q,a,b).reshape(4,4)
        out=(layers@(U[b]@rho@U[a].conj().T).ravel()).reshape(2,2)
        return np.trace(F@out)
    if q['analyze'] is None:
        def count(band,spot,s,merge):
            if spot is None:return np.ones((1,1)),0.
            e,h,g=band[spot]['herald']
            return np.ones((1,1)),e*np.sin((h+s*g)/2)**2
        pre,post=advance(q,np.ones(1),count)
        p=(pre[0]+post[0]).real
    else:p=.5*(signal(0,0)+signal(1,1)).real
    return p,(np.exp(-1j*q['phase'])*signal(0,1)).real/p

def all_nested(q,a,b):
    """Wrap the full current circuit in every later-formed pair."""
    def update(band,spot,s,merge):
        return (np.eye(16),0.) if spot is None else pair_map(band[spot],s,a,b)
    sep,common=advance(q,np.eye(4).ravel(),update)
    return sep+common

class Readout:
    def fit(self,records):
        phase=np.cos([r['input']['phase'] for r in records]);y=[r['value'] for r in records]
        self.visibility=float(np.clip(phase@y/(phase@phase),.65,.95));return self
    def predict(self,inputs):
        return self.visibility*np.array([self.fringe(q) for q in inputs],float)
