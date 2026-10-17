"""Stationary calorimeter statistics for the finite-contact rotor."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve

RATE = 3.6
FIELDS = ('angular_speed', 'flip_rate', 'temperature_a', 'temperature_b',
          'flux_fraction', 'relaxation_rate')

def _hermites(z, degree):
    z=np.asarray(z); h=np.empty((z.size,degree+1)); h[:,0]=1.
    if degree: h[:,1]=z
    for n in range(1,degree): h[:,n+1]=(z*h[:,n]-np.sqrt(n)*h[:,n-1])/np.sqrt(n+1)
    return h

@lru_cache(maxsize=16)
def _flux_collision_matrix(degree, variance):
    """Weak matrix for |v-g| times the elastic-exchange jump generator."""
    q=2*degree+4; x,xw=hermgauss(q); y,yw=laggauss(q)
    t=np.sqrt(2*variance)*x[None,:]; blocks=[]; swapped=[]; weights=[]
    for sign in (-1.,1.):
        z=sign*np.sqrt(2*variance*y[:,None]); v=(t+z)/np.sqrt(2); g=(t-z)/np.sqrt(2)
        hv=_hermites((v/np.sqrt(variance)).ravel(),degree)
        hg=_hermites((g/np.sqrt(variance)).ravel(),degree)
        blocks.append((hv[:,:,None]*hg[:,None,:]).reshape(q*q,-1))
        swapped.append((hg[:,:,None]*hv[:,None,:]).reshape(q*q,-1))
        weights.append((np.sqrt(variance)/np.pi*yw[:,None]*xw[None,:]).ravel())
    b=np.concatenate(blocks); return b.T@(np.concatenate(weights)[:,None]*(np.concatenate(swapped)-b))

def _operators(e, drag, degree):
    """Backward generator, theta velocity, and first two tilt derivatives."""
    ta,tb=float(e['temperature_a']),float(e['temperature_b'])
    omega,lam=float(e['angular_speed']),float(e['relaxation_rate'])
    eta,flip=float(e['flux_fraction']),float(e['flip_rate'])
    variance=max(ta,tb,.35)
    pairs=[(n,m) for n in range(degree+1) for m in range(degree+1) if n+m<=degree]
    small=len(pairs); ix={z:i for i,z in enumerate(pairs)}; eye=np.eye(small)
    lv=np.zeros((small,small)); lg=np.zeros_like(lv); dg=np.zeros_like(lv)
    v=np.zeros_like(lv); g=np.zeros_like(lv); g2=np.zeros_like(lv)
    for j,(n,m) in enumerate(pairs):
        lv[j,j]-=drag*n; lg[j,j]-=lam*m
        if n>=2: lv[ix[n-2,m],j]+=drag*(ta/variance-1)*np.sqrt(n*(n-1))
        if m>=2: lg[ix[n,m-2],j]+=lam*(tb/variance-1)*np.sqrt(m*(m-1))
        if n: v[ix[n-1,m],j]+=np.sqrt(variance*n)
        if (n+1,m) in ix: v[ix[n+1,m],j]+=np.sqrt(variance*(n+1))
        if m:
            g[ix[n,m-1],j]+=np.sqrt(variance*m); dg[ix[n,m-1],j]+=np.sqrt(m/variance)
        if (n,m+1) in ix: g[ix[n,m+1],j]+=np.sqrt(variance*(m+1))
        g2[j,j]+=variance*(2*m+1)
        if m>=2: g2[ix[n,m-2],j]+=variance*np.sqrt(m*(m-1))
        if (n,m+2) in ix: g2[ix[n,m+2],j]+=variance*np.sqrt((m+1)*(m+2))
    exchange=np.zeros_like(lv)
    for j,(n,m) in enumerate(pairs):
        exchange[ix[m,n],j]+=RATE*(1-eta); exchange[j,j]-=RATE*(1-eta)
    if eta:
        full=_flux_collision_matrix(degree,variance); selected=[n*(degree+1)+m for n,m in pairs]
        exchange+=RATE*eta*full[np.ix_(selected,selected)]
    plus=lv+lg+exchange+lam*omega*dg; minus=lv+lg+exchange-lam*omega*dg
    L=np.zeros((2*small,2*small)); V=np.zeros_like(L); B=np.zeros_like(L); C=np.zeros_like(L)
    L[:small,:small]=plus-flip*eye; L[small:,small:]=minus-flip*eye
    L[:small,small:]=flip*eye; L[small:,:small]=flip*eye
    V[:small,:small]=v; V[small:,small:]=v
    for off,sgn in ((0,1.),(small,-1.)):
        h2=g2-2*sgn*omega*g+omega*omega*eye
        # Shared bath Wiener term: b_Q sigma_g partial_g = -2 lambda Tb (g-u) partial_g.
        B[off:off+small,off:off+small]=lam*(h2-tb*eye)-2*lam*tb*(g@dg-sgn*omega*dg)
        C[off:off+small,off:off+small]=2*lam*tb*h2
    return L,V,B,C

def _stationary(L):
    # There is one Hermite constant per motor label.  Their *sum*, rather
    # than the + label alone, is probability normalization.
    a=L.T.copy(); a[0]=0; a[0,0]=1; a[0,len(L)//2]=1
    b=np.zeros(len(L)); b[0]=1
    return solve(a,b,check_finite=False)

def _zero_solve(L,p,b):
    a=L.astype(complex).copy(); a[0]=p; b=np.asarray(b,dtype=complex).copy(); b[0]=0
    return solve(a,b,check_finite=False)

def _statistics(e, drag):
    """Cumulant rates from the principal tilted-generator eigenvalue.

    Fourier modes remain coupled through i*n*harmonic*v.  Thus uniform initial
    angle is used correctly without incorrectly re-mixing it along a path.
    """
    eta=float(e['flux_fraction']); contrast=float(e.get('detector_contrast',0.))
    harmonic=int(e.get('detector_harmonic',1))
    if contrast and float(e['angular_speed'])!=0: raise ValueError('Angularly gated preparations require zero flow target.')
    # The relative-speed cusp converges algebraically in Hermites, so use a
    # deliberately higher order than the polynomial OU cases.  Angular
    # streaming likewise makes the gated resolvents non-polynomial.
    degree=8 if not eta and not contrast else (28 if eta else 24)
    L,V,B,C=_operators(e,float(drag),degree); p=_stationary(L); one=np.zeros(len(p)); one[0]=1; one[len(p)//2]=1
    w={0:1.} if not contrast else {0:1.,-1:contrast/2,1:contrast/2}
    w2={0:1.} if not contrast else {0:1+contrast*contrast/2,-1:contrast,1:contrast,-2:contrast*contrast/4,2:contrast*contrast/4}
    def inv(n,b):
        a=L+1j*n*harmonic*V
        return _zero_solve(a,p,b) if n==0 else solve(a,b,check_finite=False)
    mean=float(p@(B@one)); r1={}
    for n in w: r1[n]=inv(n,(mean*one if n==0 else 0)-w[n]*(B@one))
    zero=np.zeros_like(one,dtype=complex); get1=lambda n:r1.get(n,zero)
    noise=float(np.real(p@(w2.get(0,0)*(C@one)+2*sum(x*B@get1(-n) for n,x in w.items()))))
    r2={}
    for n in range(min(w2),max(w2)+1):
        bw=sum(x*(B@get1(n-j)) for j,x in w.items())
        r2[n]=inv(n,(noise*one if n==0 else 0)+2*mean*get1(n)-2*bw-w2.get(n,0)*(C@one))
    get2=lambda n:r2.get(n,zero)
    third=3*p@sum(x*(B@get2(-n))+w2.get(n,0)*(C@get1(-n)) for n,x in w.items())
    return dict(mean=mean,noise=noise,third_cumulant=float(np.real(third)))

def calibration_mean(e,drag):
    lam=float(e['relaxation_rate'])
    return drag*lam*RATE/(2*drag*lam+RATE*(drag+lam))*(float(e['temperature_a'])-float(e['temperature_b']))

def fit_drag(records):
    if not records: raise ValueError('at least one calibration record is required')
    x=np.array([(float(r['input']['temperature_a'])-float(r['input']['temperature_b']))/float(r['sigma']) for r in records])
    y=np.array([float(r['value'])/float(r['sigma']) for r in records])
    conductance=float(x@y/(x@x)); lam=float(records[0]['input']['relaxation_rate'])
    value=conductance*lam*RATE/(lam*RATE-conductance*(2*lam+RATE))
    return float(np.clip(value,.4,1.1))

@lru_cache(maxsize=2048)
def _readout(values,statistic,contrast,harmonic,drag):
    e=dict(zip(FIELDS,values))
    if statistic not in ('mean','noise','third_cumulant'): raise ValueError('unknown calorimeter statistic')
    if contrast: e.update(detector_contrast=contrast,detector_harmonic=harmonic)
    if statistic=='mean' and not contrast and e['flux_fraction']==0 and e['angular_speed']==0: return calibration_mean(e,drag)
    return _statistics(e,drag)[statistic]

def predict_at(experiments,drag):
    if drag is None: raise ValueError('call fit before predict')
    return np.asarray([_readout(tuple(float(e[f]) for f in FIELDS),e.get('statistic','mean'),float(e.get('detector_contrast',0)),int(e.get('detector_harmonic',1)),float(drag)) for e in experiments],dtype=float)

class Model:
    def __init__(self): self.drag=None
    def fit(self,records): self.drag=fit_drag(records); return self
    def predict(self,experiments): return predict_at(experiments,self.drag)
