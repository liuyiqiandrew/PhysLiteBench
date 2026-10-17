"""Spectral stationary solver for the finite-contact calorimeter."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve

RATE = 3.6

def normalized_hermites(z, degree):
    z=np.asarray(z); h=np.empty((z.size,degree+1)); h[:,0]=1.
    if degree: h[:,1]=z
    for n in range(1,degree): h[:,n+1]=(z*h[:,n]-np.sqrt(n)*h[:,n-1])/np.sqrt(n+1)
    return h

@lru_cache(maxsize=8)
def _unit_flux_collision(degree):
    """Weak matrix for |v-g| at unit Hermite variance."""
    q=2*degree+4; x,xw=hermgauss(q); y,yw=laggauss(q); t=np.sqrt(2)*x[None,:]
    values=[]; jumps=[]; weights=[]
    for sign in (-1.,1.):
        z=sign*np.sqrt(2*y[:,None]); v=(t+z)/np.sqrt(2); g=(t-z)/np.sqrt(2)
        hv=normalized_hermites(v.ravel(),degree); hg=normalized_hermites(g.ravel(),degree)
        b=(hv[:,:,None]*hg[:,None,:]).reshape(q*q,-1)
        values.append(b); jumps.append((hg[:,:,None]*hv[:,None,:]).reshape(q*q,-1)-b)
        weights.append((yw[:,None]*xw[None,:]/np.pi).ravel())
    b=np.concatenate(values)
    return b.T@(np.concatenate(weights)[:,None]*np.concatenate(jumps))

def _idx(n,m,width): return n*width+m

def _stationary_left(matrix,constant):
    system=matrix.T.copy(); rhs=np.zeros(len(constant),dtype=system.dtype)
    system[0,:]=constant; rhs[0]=1.
    return solve(system,rhs,check_finite=False)

def _poisson(matrix,p,rhs,singular):
    if not singular: return solve(matrix,rhs,check_finite=False)
    system=matrix.copy(); out=np.asarray(rhs,dtype=system.dtype).copy()
    system[0,:]=p; out[0]=0.
    return solve(system,out,check_finite=False)

def _operators(e,drag,degree):
    """Backward generator, v multiplication, and heat tilt derivatives."""
    ta=float(e['temperature_a']); tb=float(e['temperature_b']); lam=float(e['relaxation_rate'])
    eta=float(e['flux_fraction']); omega=float(e['angular_speed']); flip=float(e['flip_rate'])
    variance=max(ta,tb); width=degree+1; nvg=width*width
    # At zero target the motor label does not affect any physical variable.
    signs=(1.,) if omega==0. else (-1.,1.); ns=len(signs); size=ns*nvg
    base=np.zeros((nvg,nvg)); vop=np.zeros_like(base); gop=np.zeros_like(base)
    deriv=np.zeros_like(base); g2=np.zeros_like(base)
    for n in range(width):
      for m in range(width):
        j=_idx(n,m,width); base[j,j]-=drag*n+lam*m+RATE*(1-eta); base[_idx(m,n,width),j]+=RATE*(1-eta)
        if n>=2: base[_idx(n-2,m,width),j]+=drag*(ta/variance-1)*np.sqrt(n*(n-1))
        if m>=2: base[_idx(n,m-2,width),j]+=lam*(tb/variance-1)*np.sqrt(m*(m-1))
        if n: vop[_idx(n-1,m,width),j]+=np.sqrt(variance*n)
        if n<degree: vop[_idx(n+1,m,width),j]+=np.sqrt(variance*(n+1))
        if m:
            gop[_idx(n,m-1,width),j]+=np.sqrt(variance*m); deriv[_idx(n,m-1,width),j]+=np.sqrt(m/variance)
        if m<degree: gop[_idx(n,m+1,width),j]+=np.sqrt(variance*(m+1))
        g2[j,j]+=variance*(2*m+1)
        if m>=2:g2[_idx(n,m-2,width),j]+=variance*np.sqrt(m*(m-1))
        if m+2<width:g2[_idx(n,m+2,width),j]+=variance*np.sqrt((m+1)*(m+2))
    base+=RATE*eta*np.sqrt(variance)*_unit_flux_collision(degree)
    M=np.zeros((size,size)); V=np.zeros_like(M); G=np.zeros_like(M); D=np.zeros_like(M); G2=np.zeros_like(M); S=np.zeros_like(M)
    for a,sign in enumerate(signs):
        sl=slice(a*nvg,(a+1)*nvg)
        M[sl,sl]=base+lam*sign*omega*deriv; V[sl,sl]=vop; G[sl,sl]=gop; D[sl,sl]=deriv; G2[sl,sl]=g2; S[sl,sl]=sign*np.eye(nvg)
        if ns==2:
            M[sl,sl]-=flip*np.eye(nvg); other=slice((1-a)*nvg,(2-a)*nvg); M[sl,other]+=flip*np.eye(nvg)
    one=np.zeros(size)
    for a in range(ns):one[a*nvg]=1.
    rel=G-omega*S; rel2=G2-2*omega*S@G+omega*omega*np.eye(size)
    # Tilt is E exp(k Q): shared bath Wiener term gives -2 lambda Tb rel d_g.
    A1=lam*(rel2-tb*np.eye(size))-2*lam*tb*rel@D
    A2=2*lam*tb*rel2
    return M,V,A1,A2,one

def _cumulants(e,drag,degree=28):
    M,V,A1,A2,one=_operators(e,drag,degree); p=_stationary_left(M,one)
    mean=float(np.real(p@A1@one)); r1=_poisson(M,p,mean*one-A1@one,True)
    noise=float(np.real(p@(A2@one+2*A1@r1)))
    r2=_poisson(M,p,noise*one+2*mean*r1-2*A1@r1-A2@one,True)
    third=float(np.real(3*p@(A1@r2+A2@r1)))
    return {'mean':mean,'noise':noise,'third_cumulant':third}

def _angular_cumulants(e,drag,degree=24):
    """Fourier solve retaining theta-dot=v; modes -2..2 suffice at order 3."""
    eps=float(e.get('detector_contrast',0.)); h=int(e.get('detector_harmonic',1))
    if eps==0:return _cumulants(e,drag,degree)
    M,V,A1,A2,one=_operators(e,drag,degree); p=_stationary_left(M,one)
    w={0:1.,-1:eps/2,1:eps/2}; w2={0:1+eps*eps/2,-1:eps,1:eps,-2:eps*eps/4,2:eps*eps/4}
    blocks={n:M+1j*n*h*V for n in range(-2,3)}; mean=float(np.real(p@A1@one))
    r1={}
    for n in (-1,0,1):
        r1[n]=_poisson(blocks[n],p,(mean*one if n==0 else 0)-w.get(n,0)*A1@one,n==0)
    def wa(vectors,n):
        out=np.zeros_like(one,dtype=complex)
        for shift,weight in w.items():
            if n-shift in vectors:out+=weight*A1@vectors[n-shift]
        return out
    noise=float(np.real(p@(w2[0]*A2@one+2*wa(r1,0))))
    r2={}
    for n in range(-2,3):
        rhs=(noise*one if n==0 else 0)+2*mean*r1.get(n,0)-2*wa(r1,n)-w2.get(n,0)*A2@one
        r2[n]=_poisson(blocks[n],p,rhs,n==0)
    b2r1=np.zeros_like(one,dtype=complex)
    for shift,weight in w2.items():
        if -shift in r1:b2r1+=weight*A2@r1[-shift]
    third=float(np.real(3*p@(wa(r2,0)+b2r1)))
    return {'mean':mean,'noise':noise,'third_cumulant':third}

def calibration_mean(e,drag):
    lam=float(e['relaxation_rate'])
    return drag*lam*RATE/(2*drag*lam+RATE*(drag+lam))*(float(e['temperature_a'])-float(e['temperature_b']))

def fit_drag(records):
    if not records:raise ValueError('at least one calibration record is required')
    lam=float(records[0]['input']['relaxation_rate'])
    x=np.array([(float(r['input']['temperature_a'])-float(r['input']['temperature_b']))/float(r['sigma']) for r in records])
    y=np.array([float(r['value'])/float(r['sigma']) for r in records]); conductance=float(x@y/(x@x))
    return float(np.clip(conductance*lam*RATE/(lam*RATE-conductance*(2*lam+RATE)),.4,1.1))

@lru_cache(maxsize=1024)
def _cached_values(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction,relaxation_rate,contrast,harmonic,drag):
    e=dict(angular_speed=angular_speed,flip_rate=flip_rate,temperature_a=temperature_a,
           temperature_b=temperature_b,flux_fraction=flux_fraction,relaxation_rate=relaxation_rate,
           detector_contrast=contrast,detector_harmonic=harmonic)
    values=_angular_cumulants(e,drag) if contrast else _cumulants(e,drag)
    return values['mean'],values['noise'],values['third_cumulant']

def _readout(e,drag):
    statistic=e.get('statistic','mean')
    if statistic not in ('mean','noise','third_cumulant'):raise ValueError('unknown calorimeter statistic')
    contrast=float(e.get('detector_contrast',0.))
    if contrast:
        if float(e['angular_speed'])!=0:raise ValueError('Angularly gated preparations require zero flow target.')
    if statistic=='mean' and float(e['flux_fraction'])==0 and float(e['angular_speed'])==0:return calibration_mean(e,drag)
    values=_cached_values(float(e['angular_speed']),float(e['flip_rate']),float(e['temperature_a']),
        float(e['temperature_b']),float(e['flux_fraction']),float(e['relaxation_rate']),contrast,
        int(e.get('detector_harmonic',1)),float(drag))
    return values[{'mean':0,'noise':1,'third_cumulant':2}[statistic]]

class Model:
    def __init__(self):self.drag=None
    def fit(self,records):self.drag=fit_drag(records);return self
    def predict(self,experiments):
        if self.drag is None:raise RuntimeError('fit must be called before predict')
        return np.asarray([_readout(e,self.drag) for e in experiments],dtype=float)
