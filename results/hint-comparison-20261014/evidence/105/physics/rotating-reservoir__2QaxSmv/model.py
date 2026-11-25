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
    variance=max(e['temperature_a'],e['temperature_b']) if variance is None else float(variance)
    if e['flux_fraction']==0.:
        return hermite(e,drag,degree=6,variance=variance,total_degree=True,noise=True)
    return hermite(e,drag,degree=degree,variance=variance,noise=True)


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

def _gated_polynomial_parts(e,gamma,degree=24):
    """Polynomial operators for the zero-flow gated preparation.

    The useful state variables are x=s*v, y=s*g and alpha=s*theta.  At a
    motor flip all three variables change sign, while d alpha=x*dt between
    flips.  This change of variables removes the explicit motor state but,
    importantly, does not remove its action on the angular phase.
    """
    ta=e['temperature_a']; tb=e['temperature_b']
    lam=e['relaxation_rate']; eta=e['flux_fraction']; flip=e['flip_rate']
    variance=max(ta,tb); width=degree+1; size=width**2
    index=lambda n,m:n*width+m
    L=np.zeros((size,size)); X=np.zeros_like(L); Y=np.zeros_like(L)
    D=np.zeros_like(L); Y2=np.zeros_like(L)

    for n in range(width):
        for m in range(width):
            j=index(n,m)
            # OU generators, written in the normalized Hermite basis.
            L[j,j] -= gamma*n+lam*m+RATE*(1-eta)
            L[index(m,n),j] += RATE*(1-eta)
            if n>=2:
                L[index(n-2,m),j] += gamma*(ta/variance-1)*np.sqrt(n*(n-1))
            if m>=2:
                L[index(n,m-2),j] += lam*(tb/variance-1)*np.sqrt(m*(m-1))

            if n+1<width:
                X[index(n+1,m),j] += np.sqrt(variance*(n+1))
            if n>=1:
                X[index(n-1,m),j] += np.sqrt(variance*n)
            if m+1<width:
                Y[index(n,m+1),j] += np.sqrt(variance*(m+1))
            if m>=1:
                Y[index(n,m-1),j] += np.sqrt(variance*m)
                D[index(n,m-1),j] += np.sqrt(m/variance)

            Y2[j,j] += variance*(2*m+1)
            if m>=2:
                Y2[index(n,m-2),j] += variance*np.sqrt(m*(m-1))
            if m+2<width:
                Y2[index(n,m+2),j] += variance*np.sqrt((m+1)*(m+2))

    L += RATE*eta*collision_weak_matrix(degree,variance)
    reflection=np.diag([(-1.)**(n+m) for n in range(width) for m in range(width)])
    stationary=L+flip*(reflection-np.eye(size))

    # The two heat operators are the coefficients of z and z^2/2 in the
    # tilted generator for E exp(-z Y).
    identity=np.eye(size)
    A1=2*lam*tb*Y@D-lam*(Y2-tb*identity)
    A2=2*lam*tb*Y2

    forward=stationary.T.copy(); forward[0]=0.; forward[0,0]=1.
    target=np.zeros(size); target[0]=1.
    p=solve(forward,target,check_finite=False)
    return stationary,reflection,X,A1,A2,p,target


def _add_mode(destination,mode,value):
    if np.max(np.abs(value)) == 0:
        return
    if mode in destination:
        destination[mode] = destination[mode] + value
    else:
        destination[mode] = value.copy()


def _mode_multiply(values,operator,weights):
    result={}
    for mode,value in values.items():
        transformed=operator@value
        for shift,weight in weights.items():
            _add_mode(result,mode+shift,weight*transformed)
    return result


def angular_hermite(e,gamma=.67,degree=24,averaged=False,phase_mixing_rate=0.):
    """Exact stationary phase-coupled cumulants in a Fourier-Hermite basis.

    ``averaged`` and ``phase_mixing_rate`` remain accepted for compatibility
    with the former helper, but are intentionally not used: the physical
    phase is transported by the rotor velocity and is not an independent
    rapidly mixed label.
    """
    del averaged, phase_mixing_rate
    stationary,reflection,X,A1,A2,p,constant= _gated_polynomial_parts(e,gamma,degree)
    eps=float(e.get('detector_contrast',0.))
    harmonic=int(e.get('detector_harmonic',1))
    flip=float(e['flip_rate'])
    size=len(p)
    identity=np.eye(size)

    # Fourier modes needed through third order.  Multiplication by the gain
    # can move a mode by harmonic, and the second heat operator by twice it.
    modes=range(-2*harmonic,2*harmonic+1,harmonic)
    modes=set(modes)
    def diagonal(mode):
        return stationary-flip*(reflection-identity)+flip*(-identity)+1j*mode*X

    def solve_modes(rhs,normalized=False):
        out={}
        if normalized:
            system=stationary.astype(complex).copy()
            system[0]=p
            value=np.zeros(size,dtype=complex) if 0 not in rhs else rhs[0].astype(complex).copy()
            value[0]=0.
            out[0]=solve(system,value,check_finite=False)
        elif 0 in rhs:
            out[0]=solve(diagonal(0),rhs[0],check_finite=False)
        for mode in sorted(modes):
            if mode<=0: continue
            if mode not in rhs and -mode not in rhs: continue
            block=np.block([[diagonal(mode),flip*reflection],
                            [flip*reflection,diagonal(-mode)]])
            right=np.concatenate((rhs.get(mode,np.zeros(size,dtype=complex)),
                                  rhs.get(-mode,np.zeros(size,dtype=complex))))
            answer=solve(block,right,check_finite=False)
            out[mode]=answer[:size]; out[-mode]=answer[size:]
        return out

    w={0:1.,-harmonic:eps/2, harmonic:eps/2}
    w2={0:1+eps*eps/2,-harmonic:eps,harmonic:eps,
        -2*harmonic:eps*eps/4,2*harmonic:eps*eps/4}
    c={0:constant.astype(complex)}
    a1c=_mode_multiply(c,A1,w)
    mean=float(np.real(p@a1c[0]))
    rhs1={mode:-value for mode,value in a1c.items()}
    rhs1[0]=rhs1.get(0,np.zeros(size,dtype=complex))+mean*constant
    r1=solve_modes(rhs1,normalized=True)

    a1r1=_mode_multiply(r1,A1,w)
    a2c=_mode_multiply(c,A2,w2)
    noise=float(np.real(p@(a2c.get(0,0)+2*a1r1.get(0,0))))
    rhs2={}
    for mode in set(a1r1)|set(a2c)|set(r1):
        rhs2[mode]=2*mean*r1.get(mode,0)-2*a1r1.get(mode,0)-a2c.get(mode,0)
    rhs2[0]=rhs2.get(0,np.zeros(size,dtype=complex))+noise*constant
    r2=solve_modes(rhs2,normalized=True)

    a1r2=_mode_multiply(r2,A1,w)
    a2r1=_mode_multiply(r1,A2,w2)
    third=float(np.real(3*p@(a1r2.get(0,0)+a2r1.get(0,0))))
    return dict(mean=-mean,noise=noise,third_cumulant=-third)


@lru_cache(512)
def angular_readouts(temperature_a,temperature_b,relaxation_rate,flux_fraction,flip_rate,contrast,harmonic,drag):
    e=dict(temperature_a=temperature_a,temperature_b=temperature_b,relaxation_rate=relaxation_rate,
           flux_fraction=flux_fraction,flip_rate=flip_rate,angular_speed=0.,
           detector_contrast=contrast,detector_harmonic=harmonic)
    return angular_hermite(e,drag)

def predict_at(experiments,drag):
    result=[]
    for e in experiments:
        if e.get('detector_contrast',0.)!=0.:
            if e['angular_speed']!=0.:raise ValueError('Angularly gated preparations require zero flow target.')
            values=angular_readouts(float(e['temperature_a']),float(e['temperature_b']),float(e['relaxation_rate']),
                float(e['flux_fraction']),float(e['flip_rate']),float(e['detector_contrast']),
                int(e.get('detector_harmonic',1)),float(drag))
            result.append(values[e.get('statistic','mean')])
        else:result.append(readout(*(float(e[f]) for f in FIELDS),e.get('statistic','mean'),float(drag)))
    return np.array(result)


class Model:
    def __init__(self):self.drag=None

    def fit(self,records):
        if not records:
            raise ValueError('calibration records must not be empty')
        self.drag=fit_drag(records)
        return self

    def predict(self,experiments):
        if self.drag is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments,self.drag)
