"""Small Fock-space model of a prepared island and one pair of charge dots."""
import numpy as np
from apparatus import contacts,polarizations


def normalized(raw):
    f=np.asarray([complex(*v) for v in raw])
    return f/np.linalg.norm(f)


def generators(number):
    X=np.array([[0,1],[1,0]],complex)
    Y=np.array([[0,-1j],[1j,0]],complex)
    Z=np.diag([1.,-1.])
    modes=[]
    for j in range(number):
        for m in (X,Y):
            op=np.ones((1,1))
            for k in range(number):
                op=np.kron(op,Z if k<j else m if k==j else np.eye(2))
            modes.append(op)
    return np.array(modes)


class Island:
    def __init__(self,q):
        used={j for arm in q['probe'] for a,b,_ in arm for j in (a,b)}
        used.update(j for encounter in q['encounters'] for _,a,b,_,_,_ in encounter['pulses']
                    for j in (a,b))
        if not used:used.update((0,1))
        if len(used)%2:
            used.add(next(j for j in range(2*q['n']) if j not in used))
        ids=sorted(used)
        modes=generators(len(ids)//2)
        self.sites=dict(zip(ids,modes))
        self.identity=np.eye(len(modes[0]))
        frame=contacts(q)[ids]
        polar=polarizations(q)
        def density(t):
            half=(frame[:,::2]*t)@frame[:,1::2].T
            vals,vec=np.linalg.eigh(1j*(half-half.T))
            rho=self.identity.astype(complex)/len(self.identity)
            for j in np.flatnonzero(vals>1e-12):
                a=np.einsum('a,aij->ij',np.sqrt(2)*vec[:,j].real,modes)
                b=np.einsum('a,aij->ij',-np.sqrt(2)*vec[:,j].imag,modes)
                rho=rho@(self.identity-1j*vals[j]*a@b)
            return rho
        weight=q['parity']*np.prod(polar)
        self.rho=(density(polar)+weight*density(1/polar))/(1+weight)
        dots=generators(2)
        self.lower=[np.kron((dots[2*j]+1j*dots[2*j+1])/2,self.identity) for j in range(2)]
        self.parity=np.diag([1.,-1.,-1.,1.])

    def outgoing(self,encounter,counter_value):
        """Return the island operators for exit in dot 0 or dot 1."""
        initial=normalized(encounter['prepare'])
        size=len(self.identity)
        state=np.kron(np.array([0,initial[1],initial[0],0])[:,None],self.identity)
        for j,a,b,t,phase,hold in encounter['pulses']:
            phase+=counter_value*encounter['feedback']*j
            level=np.kron(self.parity,(self.sites[a]+1j*self.sites[b])/2)
            hop=np.exp(1j*phase)*self.lower[j].conj().T@level
            hop+=hop.conj().T.copy()
            advanced=hop@state
            state+=-1j*np.sin(t)*advanced+(np.cos(t)-1)*(hop@advanced)
            state[3*size:]*=np.exp(-1j*hold)
        return state.reshape(4,size,size)[[2,1]]

    def probe(self,arms,rho,angle):
        operators=[]
        for pulses in arms:
            op=self.identity
            for a,b,t in pulses:
                op=(np.cos(t/2)*self.identity-np.sin(t/2)*self.sites[a]@self.sites[b])@op
            operators.append(op)
        return (np.exp(-1j*angle)*np.trace(operators[0].conj().T@operators[1]@rho)).real/np.trace(rho).real
