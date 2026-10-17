"""Finite-contact kinetic moments and continuous calorimeter statistics."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve

RATE=3.6

def normalized_hermites(z, degree):
    z=np.asarray(z);h=np.zeros((z.size,degree+1));h[:,0]=1.
    if degree:h[:,1]=z
    for n in range(1,degree):h[:,n+1]=(z*h[:,n]-np.sqrt(n)*h[:,n-1])/np.sqrt(n+1)
    return h


@lru_cache(16)
def collision_weak_matrix(degree, variance=1.6, quadrature=None):
    """Exact Gaussian polynomial integral for the nonsmooth collision rate.

    Rotate to z=(v-g)/sqrt(2),t=(v+g)/sqrt(2). Gauss-Laguerre integrates
    |z| times its Gaussian exactly after z=+/-sqrt(2*variance*y).
    Summing both signs cancels odd z powers; Hermite integrates the t part.
    """
    q=quadrature or (2*degree+4)
    th,tw=hermgauss(q);yl,yw=laggauss(q)
    t=np.sqrt(2*variance)*th[None,:]
    blocks=[];swapped=[];weights=[]
    for sign in [-1.,1.]:
        z=sign*np.sqrt(2*variance*yl[:,None])
        v=(t+z)/np.sqrt(2);g=(t-z)/np.sqrt(2)
        hv=normalized_hermites(v.ravel()/np.sqrt(variance),degree)
        hg=normalized_hermites(g.ravel()/np.sqrt(variance),degree)
        blocks.append((hv[:,:,None]*hg[:,None,:]).reshape(q*q,-1))
        swapped.append((hg[:,:,None]*hv[:,None,:]).reshape(q*q,-1))
        weights.append((np.sqrt(variance)/np.pi*yw[:,None]*tw[None,:]).ravel())
    b=np.concatenate(blocks);jump=np.concatenate(swapped)-b;w=np.concatenate(weights)
    matrix=b.T@(w[:,None]*jump)
    # The reversible swap generator is symmetric and dissipative under the
    # isotropic Gaussian integration weight, including its |v-g| rate.
    assert np.max(abs(matrix-matrix.T))<1e-9
    return matrix


def hermite(e, gamma=.67, degree=12, variance=1.6, quadrature=None, total_degree=False, noise=False):
    """Independent polynomial weak generator; no FV or Gaussian closure calls."""
    ta=e['temperature_a'];tb=e['temperature_b'];omega=e['angular_speed']
    lam=e['relaxation_rate'];eta=e['flux_fraction'];flip=e['flip_rate']
    width=degree+1;size=width**2;matrix=np.zeros((size,size))
    index=lambda n,m:n*width+m
    for n in range(width):
        for m in range(width):
            j=index(n,m)
            matrix[j,j]-=gamma*n+lam*m+flip*(1-(-1)**(n+m))+RATE*(1-eta)
            matrix[index(m,n),j]+=RATE*(1-eta)
            if n>=2:matrix[index(n-2,m),j]+=gamma*(ta/variance-1)*np.sqrt(n*(n-1))
            if m>=2:matrix[index(n,m-2),j]+=lam*(tb/variance-1)*np.sqrt(m*(m-1))
            if m>=1:matrix[index(n,m-1),j]+=lam*omega/np.sqrt(variance)*np.sqrt(m)
    matrix+=RATE*eta*collision_weak_matrix(degree,variance,quadrature)
    retained=np.array([index(n,m) for n in range(width) for m in range(width) if not total_degree or n+m<=degree])
    lookup={j:i for i,j in enumerate(retained)}
    matrix=matrix[np.ix_(retained,retained)];size=len(retained)
    index_full=index
    index=lambda n,m:lookup[index_full(n,m)]
    forward=matrix.T.copy();forward[0]=0.;forward[0,0]=1.;rhs=np.zeros(size);rhs[0]=1.
    moments=solve(forward,rhs,check_finite=False)
    mv=np.sqrt(variance)*moments[index(1,0)];mg=np.sqrt(variance)*moments[index(0,1)]
    v2=variance*(1+np.sqrt(2)*moments[index(2,0)]);g2=variance*(1+np.sqrt(2)*moments[index(0,2)])
    vg=variance*moments[index(1,1)]
    mean=lam*(g2-2*omega*mg+omega*omega-tb)
    v2g2=variance**2*(1+np.sqrt(2)*(moments[index(2,0)]+moments[index(0,2)])+2*moments[index(2,2)])
    result=dict(mean=float(mean),vmean=float(mv),gmean=float(mg),v2=float(v2),g2=float(g2),vg=float(vg),
                energy_covariance=float(.25*(v2g2-v2*g2)),
                bath_a=float(gamma*(v2-ta)),motor=float(lam*(omega*omega-omega*mg)),
                first_law_residual=float(abs(gamma*(v2-ta)+mean-lam*(omega*omega-omega*mg))),
                stationary_residual=float(max(abs(matrix.T@moments))))
    if noise:
        # Independent continuous diffusion Feynman--Kac derivatives. No
        # finite-volume heat marks, stochastic paths or shot-noise closure.
        # dQ=lam[(g-u)^2-Tb]dt-sqrt(2lamTb)(g-u)dW_b.
        derivative=np.zeros((width**2,width**2));multiply=np.zeros_like(derivative);square=np.zeros_like(derivative)
        for n in range(width):
            for m in range(width):
                j=index_full(n,m)
                square[j,j]=variance*(2*m+1)
                if m>=1:
                    derivative[index_full(n,m-1),j]=np.sqrt(m/variance)
                    multiply[index_full(n,m-1),j]=np.sqrt(variance*m)
                if m+1<width:multiply[index_full(n,m+1),j]=np.sqrt(variance*(m+1))
                if m>=2:square[index_full(n,m-2),j]=variance*np.sqrt(m*(m-1))
                if m+2<width:square[index_full(n,m+2),j]=variance*np.sqrt((m+1)*(m+2))
        identity=np.eye(width**2);relative_square=square-2*omega*multiply+omega**2*identity
        first=2*lam*tb*(multiply-omega*identity)@derivative-lam*(relative_square-tb*identity)
        second=2*lam*tb*relative_square
        first=first[np.ix_(retained,retained)];second=second[np.ix_(retained,retained)]
        drift=-first[:,0];constant=np.zeros(size);constant[0]=1.
        tilted_mean=float(moments@first[:,0]);system=matrix.copy();system[0]=moments
        rhs=tilted_mean*constant-first[:,0];rhs[0]=0.
        r1=solve(system,rhs,check_finite=False)
        shot=float(moments@second[:,0]);correlation=float(2*moments@first@r1)
        result.update(noise=shot+correlation,noise_shot=shot,noise_correlation=correlation,
                      noise_poisson_residual=float(max(abs(matrix@r1-(tilted_mean*constant-first[:,0])))))
        # Third derivative of the principal eigenvalue of L(k) for exp(-k Q).
        # p r1 = p r2 = 0 fixes the response normalization.
        cumulant2=shot+correlation
        rhs2=cumulant2*constant+2*tilted_mean*r1-2*first@r1-second[:,0]
        rhs2[0]=0.
        r2=solve(system,rhs2,check_finite=False)
        cumulant3_minus=float(moments@(3*first@r2+3*second@r1))
        result.update(third_cumulant=-cumulant3_minus,
                      third_poisson_residual=float(max(abs(matrix@r2-(cumulant2*constant+2*tilted_mean*r1-2*first@r1-second[:,0])))))
    return result


def kinetic_statistics(e,drag,degree=34,variance=None):
    """Cumulants from the backward generator, retaining both motor states.

    The old implementation folded the two motor labels into a parity term.
    That is not equivalent once the target flow is nonzero.  Keeping the
    labels explicitly also makes the flip-rate dependence physical.
    """
    ta,tb=e['temperature_a'],e['temperature_b']
    lam,eta,flip=e['relaxation_rate'],e['flux_fraction'],e['flip_rate']
    omega=e['angular_speed']; variance=max(ta,tb) if variance is None else float(variance)
    # Degree 22 is a useful compromise for the |v-g| collision kernel.  The
    # no-flux process is quadratic and needs only a small basis.
    degree=8 if eta == 0. else min(int(degree),22)
    w=degree+1; base=w*w; size=2*base
    ix=lambda s,n,m: s*base+n*w+m
    L=np.zeros((size,size)); A1=np.zeros_like(L); A2=np.zeros_like(L)
    coll=collision_weak_matrix(degree,variance)
    for s in range(2):
        sign=-1. if s==0 else 1.
        u=sign*omega
        for n in range(w):
            for m in range(w):
                j=ix(s,n,m)
                L[j,j]-=drag*n+lam*m+RATE*(1-eta)+flip
                L[ix(1-s,n,m),j]+=flip
                L[ix(s,m,n),j]+=RATE*(1-eta)
                if n>=2: L[ix(s,n-2,m),j]+=drag*(ta/variance-1)*np.sqrt(n*(n-1))
                if m>=2: L[ix(s,n,m-2),j]+=lam*(tb/variance-1)*np.sqrt(m*(m-1))
                if m>=1:
                    L[ix(s,n,m-1),j]+=lam*u*np.sqrt(m/variance)
                # Tilted heat generator for E exp(-k Q).
                # A1 includes the shared B-noise cross derivative.
                A1[j,j]+= -lam*(variance*(2*m+1)+u*u-tb)
                if m>=1:
                    A1[ix(s,n,m-1),j]+=2*lam*tb*np.sqrt(m/variance)+2*lam*u*np.sqrt(variance*m)
                if m+1<w: A1[ix(s,n,m+1),j]+=-lam*2*u*np.sqrt(variance*(m+1))
                if m>=2: A1[ix(s,n,m-2),j]+=-lam*variance*np.sqrt(m*(m-1))
                if m+2<w: A1[ix(s,n,m+2),j]+=-lam*variance*np.sqrt((m+1)*(m+2))
                A2[j,j]+=2*lam*tb*(variance*(2*m+1)+u*u)
                if m>=1: A2[ix(s,n,m-1),j]+=-4*lam*tb*u*np.sqrt(variance*m)
                if m+1<w: A2[ix(s,n,m+1),j]+=-4*lam*tb*u*np.sqrt(variance*(m+1))
                if m>=2: A2[ix(s,n,m-2),j]+=2*lam*tb*variance*np.sqrt(m*(m-1))
                if m+2<w: A2[ix(s,n,m+2),j]+=2*lam*tb*variance*np.sqrt((m+1)*(m+2))
        sl=slice(s*base,(s+1)*base)
        L[sl,sl]+=RATE*eta*coll
    # Construct polynomial multiplication/derivative operators explicitly.
    # Doing this as products is important: g*d/dg has both diagonal and
    # degree-minus-two Hermite components.
    G=np.zeros((base,base)); D=np.zeros_like(G); G2=np.zeros_like(G)
    b=lambda n,m:n*w+m
    for n in range(w):
        for m in range(w):
            j=b(n,m)
            if m: G[b(n,m-1),j]+=np.sqrt(variance*m); D[b(n,m-1),j]+=np.sqrt(m/variance)
            if m+1<w: G[b(n,m+1),j]+=np.sqrt(variance*(m+1))
            G2[j,j]=variance*(2*m+1)
            if m>=2: G2[b(n,m-2),j]=variance*np.sqrt(m*(m-1))
            if m+2<w: G2[b(n,m+2),j]=variance*np.sqrt((m+1)*(m+2))
    A1.fill(0.); A2.fill(0.); eye=np.eye(base)
    for s in range(2):
        u=(-1. if s==0 else 1.)*omega; rel2=G2-2*u*G+u*u*eye
        sl=slice(s*base,(s+1)*base)
        A1[sl,sl]=2*lam*tb*(G-u*eye)@D-lam*(rel2-tb*eye)
        A2[sl,sl]=2*lam*tb*rel2
    # Stationary expectation functional p satisfies L^T p=0.
    p_system=L.T.copy(); rhs=np.zeros(size); c=np.zeros(size); c[ix(0,0,0)]=c[ix(1,0,0)]=1.
    p_system[0]=c; rhs[0]=1.
    p=solve(p_system,rhs,check_finite=False)
    a1=A1@c; mean_minus=float(p@a1); mean=-mean_minus
    # A rapidly independently mixed gate only changes equal-time quadratic
    # variation.  It is the limiting preparation documented in the README.
    contrast=float(e.get('detector_contrast',0.))
    gate2=1.+contrast*contrast/2.
    A2*=gate2; a2=A2@c
    system=L.copy(); system[0]=p
    r_rhs=mean_minus*c-a1; r_rhs[0]=0.
    r1=solve(system,r_rhs,check_finite=False)
    noise=float(p@a2+2*p@(A1@r1))
    r2_rhs=noise*c+2*mean_minus*r1-2*A1@r1-a2; r2_rhs[0]=0.
    r2=solve(system,r2_rhs,check_finite=False)
    third_minus=float(p@(3*A1@r2+3*A2@r1))
    return dict(mean=mean,noise=noise,third_cumulant=-third_minus)


@lru_cache(4096)
def readout(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction,relaxation_rate,statistic,drag):
    e=dict(zip(FIELDS,(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction,relaxation_rate)))
    if statistic not in ['mean','noise','third_cumulant']:raise ValueError('unknown calorimeter statistic')
    if statistic=='mean' and flux_fraction==0. and angular_speed==0.:
        return float(calibration_mean(e,drag))
    return kinetic_statistics(e,drag)[statistic]


RATE=3.6
FIELDS=('angular_speed','flip_rate','temperature_a','temperature_b','flux_fraction','relaxation_rate')


def calibration_mean(e,drag):
    lam=e['relaxation_rate']
    return drag*lam*RATE/(2*drag*lam+RATE*(drag+lam))*(e['temperature_a']-e['temperature_b'])


def fit_drag(records):
    lam=float(records[0]['input']['relaxation_rate'])
    assert all(r['input']['relaxation_rate']==lam for r in records)
    contrast=np.array([(r['input']['temperature_a']-r['input']['temperature_b'])/r['sigma'] for r in records])
    measured=np.array([r['value']/r['sigma'] for r in records])
    conductance=float(contrast@measured/(contrast@contrast))
    return float(np.clip(conductance*lam*RATE/(lam*RATE-conductance*(2*lam+RATE)),.4,1.1))


def hermite_operators(e,gamma,degree=32):
    ta=e['temperature_a'];tb=e['temperature_b'];lam=e['relaxation_rate'];eta=e['flux_fraction'];variance=max(ta,tb)
    width=degree+1;size=width**2;index=lambda n,m:n*width+m;L=np.zeros((size,size));V=np.zeros_like(L);G=np.zeros_like(L);dg=np.zeros_like(L);G2=np.zeros_like(L)
    assert e['angular_speed']==0.
    for n in range(width):
        for m in range(width):
            j=index(n,m);L[j,j]-=gamma*n+lam*m+3.6*(1-eta);L[index(m,n),j]+=3.6*(1-eta)
            if n>=2:L[index(n-2,m),j]+=gamma*(ta/variance-1)*np.sqrt(n*(n-1))
            if m>=2:L[index(n,m-2),j]+=lam*(tb/variance-1)*np.sqrt(m*(m-1))
            if n>=1:V[index(n-1,m),j]=np.sqrt(variance*n)
            if n<degree:V[index(n+1,m),j]=np.sqrt(variance*(n+1))
            if m>=1:
                G[index(n,m-1),j]=np.sqrt(variance*m);dg[index(n,m-1),j]=np.sqrt(m/variance)
            if m<degree:G[index(n,m+1),j]=np.sqrt(variance*(m+1))
            G2[j,j]=variance*(2*m+1)
            if m>=2:G2[index(n,m-2),j]=variance*np.sqrt(m*(m-1))
            if m+2<width:G2[index(n,m+2),j]=variance*np.sqrt((m+1)*(m+2))
    L+=3.6*eta*collision_weak_matrix(degree,variance)
    A1=2*lam*tb*G@dg-lam*(G2-tb*np.eye(size));A2=2*lam*tb*G2
    system=L.T.copy();system[0]=0.;system[0,0]=1.;target=np.zeros(size);target[0]=1.;p=solve(system,target,check_finite=False)
    constant=np.zeros(size);constant[0]=1.
    return L,V,A1,A2,p,constant

def angular_hermite(e,gamma=.67,degree=32,averaged=False,phase_mixing_rate=0.):
    L,V,A1,A2,p,constant=hermite_operators(e,gamma,degree)
    eps=e.get('detector_contrast',0.);harmonic=e.get('detector_harmonic',1)
    w={0:1.,-1:eps/2,1:eps/2};w2={0:1+eps*eps/2,-1:eps,1:eps,-2:eps*eps/4,2:eps*eps/4}
    ns=[0];mean=p@A1@constant
    def invert(n,rhs):
        if n:
            return solve(L+1j*n*harmonic*V-phase_mixing_rate*(n*harmonic)**2*np.eye(len(p)),rhs,check_finite=False)
        system=L.astype(complex);system[0]=p;rhs=rhs.copy();rhs[0]=0.;return solve(system,rhs,check_finite=False)
    r1={n:invert(n,(mean*constant if n==0 else np.zeros(len(p)))-w[n]*(A1@constant)) for n in ns}
    c2=float(np.real(p@(w2[0]*(A2@constant)+2*sum(w[-n]*(A1@r1[n]) for n in ns))))
    r2={n:invert(n,(c2*constant if n==0 else np.zeros(len(p)))+2*mean*r1[n]-2*sum(w.get(n-m,0.)*(A1@r1[m]) for m in ns)-w2[n]*(A2@constant)) for n in ns}
    c3=float(np.real(3*p@sum(w[-n]*(A1@r2[n])+w2[-n]*(A2@r1[n]) for n in ns)))
    return dict(mean=-float(mean),noise=c2,third_cumulant=-c3)


@lru_cache(512)
def angular_readouts(temperature_a,temperature_b,relaxation_rate,flux_fraction,contrast,harmonic,drag):
    e=dict(temperature_a=temperature_a,temperature_b=temperature_b,relaxation_rate=relaxation_rate,
           flux_fraction=flux_fraction,angular_speed=0.,detector_contrast=contrast,detector_harmonic=harmonic)
    return angular_hermite(e,drag)

def predict_at(experiments,drag):
    if drag is None:
        raise RuntimeError('Call fit(records) before predict(experiments).')
    result=[]
    for e in experiments:
        if e.get('detector_contrast',0.)!=0.:
            if e['angular_speed']!=0.:raise ValueError('Angularly gated preparations require zero flow target.')
            statistic=e.get('statistic','mean')
            if statistic not in ('mean','noise','third_cumulant'): raise ValueError('unknown calorimeter statistic')
            result.append(kinetic_statistics(e,float(drag))[statistic])
        else:result.append(readout(*(float(e[f]) for f in FIELDS),e.get('statistic','mean'),float(drag)))
    return np.array(result)


class Model:
    def __init__(self):self.drag=None

    def fit(self,records):
        if not records:
            raise ValueError('at least one calibration record is required')
        self.drag=fit_drag(records)
        return self

    def predict(self,experiments):return predict_at(experiments,self.drag)
