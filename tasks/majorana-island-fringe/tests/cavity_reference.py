"""Direct spin density matrices and coherent-state quadrature of the optical network."""
import numpy as np
from itertools import product
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.fft import dct


def active_density(q):
    labels = {j for arm in q["arms"] for a,b,_,_ in arm for j in (a,b)}
    labels.update(j for bins in q["cavity"]["segments"] for _,a,b,_,_ in bins for j in (a,b))
    if len(labels)%2:
        labels.add(next(i for i in range(2*q["n"]) if i not in labels))
    ix = sorted(labels)
    count = len(ix)//2
    I, Z = np.eye(2),np.diag([1.,-1.])
    majorana=[]
    for j in range(count):
        for op in (np.array([[0.,1.],[1.,0.]]), np.array([[0.,-1j],[1j,0.]])):
            m = np.ones((1,1))
            for k in range(count):
                m = np.kron(m, Z if k<j else op if k==j else I)
            majorana.append(m)

    n=q["n"]
    basis=(np.eye(2*n) if q["frame"]=="site" else dct(np.eye(2*n),norm="ortho",axis=0))[ix]
    t=np.tanh(q["beta"]*(1+q["ripple"]*np.cos((2*np.arange(n)+1)*np.pi/n))/2)
    parity=q["parity"]*np.prod(t)
    dim=2**count
    density=np.zeros((dim,dim),complex)
    for temp,weight in [(t,1.),(1/t,parity)]:
        h=(basis[:,0::2]*temp)@basis[:,1::2].T
        cov=1j*(h-h.T)
        moment={0:1.}
        for mask in range(1<<len(ix)):
            if mask.bit_count()%2:
                continue
            indices=[i for i in range(len(ix)) if mask>>i&1]
            if indices:
                a=indices[0]
                moment[mask]=sum((-1)**i*cov[a,b]*moment[mask^(1<<a)^(1<<b)]
                                 for i,b in enumerate(indices[1:]))
            op=np.eye(dim)
            for j in indices:
                op=op@majorana[j]
            density+=weight*moment[mask]*op.conj().T
    return density/(dim*(1+parity)), dict(zip(ix,majorana))


def reference(q, thermal_order=12, window_order=20):
    c=q["cavity"]
    rho, gamma=active_density(q)
    ident=np.eye(len(rho))
    gh, weights=hermgauss(thermal_order)
    z0=(np.sqrt(c["thermal"])*(gh[:,None]+1j*gh)).ravel()
    weights=(weights[:,None]*weights).ravel()/np.pi
    lo,hi=c["window"]; x,wx=leggauss(window_order)
    x=(hi+lo)/2+(hi-lo)*x/2; wx *= (hi-lo)/2
    mode=np.r_[0j,[complex(*r) for r in c["mode"]]]
    mode/=np.linalg.norm(mode)
    def states(arm):
        outputs=[]
        for meter in (1,-1):
            for labels in product((1,-1),repeat=len(c["transmission"])):
                z=np.zeros((len(z0),len(labels)+1),complex)
                z[:,0]=z0
                phase=np.zeros(len(z0))
                spin=ident.copy()
                start=0
                for j,(e,a,b,ax,ay) in enumerate(c["segments"][arm]):
                    for i,k,th,g in q["arms"][arm][start:e]:
                        half=(th+meter*g)/2
                        spin=(np.cos(half)*ident-np.sin(half)*gamma[i]@gamma[k])@spin
                    start=e; sig=labels[j]
                    spin=(ident-1j*sig*gamma[a]@gamma[b])@spin/2
                    kick=sig*complex(ax,ay)
                    phase+=np.imag(z[:,0].conj()*kick)
                    z[:,0]+=kick
                    eta=c["transmission"][j]
                    z[:,j+1]=-np.sqrt(1-eta)*z[:,0]
                    z[:,0]*=np.sqrt(eta)
                if meter==-1:
                    spin=spin*(q["outcome"]*np.exp(-1j*q["meter_phase"]))
                proj=z@mode.conj()
                rest=z-proj[:,None]*mode
                proj*=np.exp(-1j*c["angle"])
                center=np.sqrt(2)*proj.real[:,None]
                momentum=np.sqrt(2)*proj.imag[:,None]
                psi=np.pi**(-.25)*np.exp(-.5*(x-center)**2+1j*momentum*(x-center/2))
                outputs.append((spin/2,rest,np.exp(1j*phase[:,None])*psi))
        return outputs
    def overlap(left,right):
        total=0j
        for a,za,pa in left:
            for b,zb,pb in right:
                spin=np.trace(a.conj().T@b@rho)
                if abs(spin)<1e-16:
                    continue
                rest=np.exp(-.5*np.sum(abs(za-zb)**2,axis=1)+1j*np.sum(za.conj()*zb,axis=1).imag)
                total+=spin*np.dot(weights*rest,(pa.conj()*pb)@wx)
        return total
    a,b=states(0),states(1)
    accept=(overlap(a,a)+overlap(b,b)).real/2
    return accept,(np.exp(-1j*q["phase"])*overlap(a,b)).real/accept
