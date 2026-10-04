"""Independent conservative probability-current and molecular force reference."""
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

TRUE_PARAMETER = 4.1

def bernoulli(x):
    x=np.asarray(x)
    return np.divide(x, np.expm1(x), out=np.ones_like(x), where=abs(x)>1e-10)


def finite_volume(extension, rotation, temperature, drag, length, nr=80, nt=128):
    """Positive conservative polar SG balance, using actual probability masses."""
    dr=length/nr; dt=2*np.pi/nt
    r=(np.arange(nr)+.5)*dr; theta=(np.arange(nt)+.5)*dt
    volume=np.repeat(r*dr*dt,nt)
    ids=np.arange(nr*nt).reshape(nr,nt)
    diffusion=2*temperature/drag
    left=ids[:-1].ravel(); right=ids[1:].ravel()
    edge=diffusion*(r[:-1]+dr/2)*dt/dr
    pe=extension*(r[1:,None]**2-r[:-1,None]**2)*np.cos(2*theta)/(2*diffusion)
    pe+=length**2/(2*temperature)*np.log((1-r[1:,None]**2/length**2)/(1-r[:-1,None]**2/length**2))
    ab=np.broadcast_to(edge[:,None],pe.shape).ravel()*bernoulli(-pe.ravel())/volume[left]
    ba=np.broadcast_to(edge[:,None],pe.shape).ravel()*bernoulli(pe.ravel())/volume[right]
    l2=ids.ravel();r2=np.roll(ids,-1,axis=1).ravel()
    edge2=diffusion*dr/(r[:,None]*dt)
    pe2=r[:,None]**2/diffusion*(rotation*dt+extension/2*(np.cos(2*(theta+dt))-np.cos(2*theta)))
    ab2=(edge2*bernoulli(-pe2)).ravel()/volume[l2]
    ba2=(edge2*bernoulli(pe2)).ravel()/volume[r2]
    left=np.r_[left,l2];right=np.r_[right,r2];ab=np.r_[ab,ab2];ba=np.r_[ba,ba2]
    a=coo_matrix((np.r_[-ab,ab,ba,-ba],(np.r_[left,right,left,right],np.r_[left,left,right,right])),shape=(nr*nt,nr*nt)).tocsc()
    fixed=a[1:,1:];rhs=-a[1:,0].toarray().ravel()
    p=np.r_[1.,spsolve(fixed,rhs)];p/=p.sum()
    rr,tt=np.meshgrid(r,theta,indexing='ij')
    force=np.dot(p,(rr**2*np.cos(2*tt)/(1-rr**2/length**2)).ravel())
    cx=np.dot(p,(rr**2*np.cos(tt)**2).ravel());cy=np.dot(p,(rr**2*np.sin(tt)**2).ravel())
    cxy=np.dot(p,(rr**2*np.cos(tt)*np.sin(tt)).ravel());c=np.array([[cx,cxy],[cxy,cy]])
    stress=drag/2*(extension*np.trace(c)-2*rotation*cxy)
    current=ab*p[left]-ba*p[right]
    return float(force), c, {'moment_stress':float(stress),'force_moment_error':float(abs(force-stress)),
        'minimum_probability':float(p.min()),'conservation':float(np.max(abs(a@p))),
        'current_l1':float(np.sum(abs(current)))}


def experiment(rate, rotation, temperature, max_length):
    return dict(rate=float(rate), rotation=float(rotation), temperature=float(temperature),
                max_length=None if max_length is None else float(max_length))


def calibration_inputs():
    settings = [experiment(r,o,t,None) for r in [-.22,-.12,.08,.2]
                for o in [-.35,0.,.3] for t in [.8,1.2]]
    return settings*6


def hidden_inputs():
    return {
      'hookean_anchors':[experiment(r,o,t,None) for r,o,t in [(.24,.65,1.),(-.21,-.8,.9),(.12,.2,1.1)]],
      'mixed_extension':[experiment(*args) for args in [(.65,.2,.8,np.sqrt(6)),(.8,.35,1.,3.),(.95,.5,1.2,np.sqrt(12))]],
      'rapid_rotation':[experiment(*args) for args in [(.8,1.1,.8,np.sqrt(6)),(.65,1.,1.,3.),(.6,1.2,1.,3.)]],
      'signed_flows':[experiment(*args) for args in [(-.75,-.3,1.1,2.8),(.7,-.45,.9,3.2),(-.85,.4,1.,3.1)]],
    }


@lru_cache(maxsize=4096)
def reading(rate, rotation, temperature, drag, length):
    if length is None:
        # Direct solution of the two-bead Ornstein-Uhlenbeck covariance equations.
        return temperature*drag*rate/(1+drag**2*(rotation**2-rate**2)/4)
    coarse = finite_volume(rate,rotation,temperature,drag,length,nr=64,nt=96)[0]
    fine = finite_volume(rate,rotation,temperature,drag,length,nr=128,nt=192)[0]
    return (4*fine-coarse)/3


def predict(experiments, drag=TRUE_PARAMETER):
    return np.array([reading(e['rate'],e['rotation'],e['temperature'],drag,e['max_length']) for e in experiments])
