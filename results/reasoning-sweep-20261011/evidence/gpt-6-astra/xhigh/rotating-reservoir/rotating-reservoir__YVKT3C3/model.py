"""Finite-contact kinetic moments and continuous calorimeter statistics."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import lu_factor, lu_solve, solve

RATE=3.6

def normalized_hermites(z, degree):
    z=np.asarray(z);h=np.zeros((z.size,degree+1));h[:,0]=1.
    if degree:h[:,1]=z
    for n in range(1,degree):h[:,n+1]=(z*h[:,n]-np.sqrt(n)*h[:,n-1])/np.sqrt(n+1)
    return h


def collision_weak_matrix(degree, variance=1.6, quadrature=None):
    # In normalized Hermite coordinates only the speed in the collision
    # hazard depends on the reference variance. Cache the unit-scale matrix.
    return np.sqrt(variance)*_unit_collision_weak_matrix(degree, quadrature)


@lru_cache(8)
def _unit_collision_weak_matrix(degree, quadrature=None):
    """Exact Gaussian polynomial integral for the nonsmooth collision rate.

    Rotate to z=(v-g)/sqrt(2),t=(v+g)/sqrt(2). Gauss-Laguerre integrates
    |z| times its Gaussian exactly after z=+/-sqrt(2*variance*y).
    Summing both signs cancels odd z powers; Hermite integrates the t part.
    """
    variance=1.
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
    if eta:
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
        factor=lu_factor(system,check_finite=False)
        rhs=tilted_mean*constant-first[:,0];rhs[0]=0.
        r1=lu_solve(factor,rhs,check_finite=False)
        shot=float(moments@second[:,0]);correlation=float(2*moments@first@r1)
        result.update(noise=shot+correlation,noise_shot=shot,noise_correlation=correlation,
                      noise_poisson_residual=float(max(abs(matrix@r1-(tilted_mean*constant-first[:,0])))))
        # Third derivative of the principal eigenvalue of L(k) for exp(-k Q).
        # p r1 = p r2 = 0 fixes the response normalization.
        cumulant2=shot+correlation
        rhs2=cumulant2*constant+2*tilted_mean*r1-2*first@r1-second[:,0]
        rhs2[0]=0.
        r2=lu_solve(factor,rhs2,check_finite=False)
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
    return ungated_readouts(angular_speed,flip_rate,temperature_a,temperature_b,
                           flux_fraction,relaxation_rate,drag)[statistic]


@lru_cache(512)
def ungated_readouts(angular_speed,flip_rate,temperature_a,temperature_b,
                    flux_fraction,relaxation_rate,drag):
    e=dict(zip(FIELDS,(angular_speed,flip_rate,temperature_a,temperature_b,
                       flux_fraction,relaxation_rate)))
    return kinetic_statistics(e,drag)


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
    if eta:
        L+=RATE*eta*collision_weak_matrix(degree,variance)
    A1=2*lam*tb*G@dg-lam*(G2-tb*np.eye(size));A2=2*lam*tb*G2
    system=L.T.copy();system[0]=0.;system[0,0]=1.;target=np.zeros(size);target[0]=1.;p=solve(system,target,check_finite=False)
    constant=np.zeros(size);constant[0]=1.
    return L,V,A1,A2,p,constant

def angular_hermite(e,gamma=.67,degree=32,averaged=False,phase_mixing_rate=0.):
    """Cumulant rates from the joint angle/velocity tilted generator.

    On Fourier mode n, physical angle transport contributes i*n*h*V.
    For exp(-k Y), the tilt is k*w*A1 + k**2*w**2*A2/2; A1 includes
    the cross derivative from the shared bath-B Wiener noise.
    The first eigenvector response has modes 0, +/-1. Although the second
    response also has +/-2, only 0, +/-1 can contribute to the third
    eigenvalue derivative. Negative modes are conjugates of positive ones.

    The optional averaged flag is a diagnostic rapid-phase approximation;
    predictions use the actual angle dynamics with no extra phase diffusion.
    """
    L,V,A1,A2,p,constant=hermite_operators(e,gamma,degree)
    eps=e.get('detector_contrast',0.);harmonic=e.get('detector_harmonic',1)
    b1=A1[:,0];b2=A2[:,0];mean=float(p@b1)
    w2_zero=1.+eps*eps/2

    # The stationary normalization p*r=0 removes the zero-mode nullspace.
    system=L.copy();system[0]=p
    zero_factor=lu_factor(system,check_finite=False)
    def invert_zero(rhs):
        rhs=rhs.copy();rhs[0]=0.
        return lu_solve(zero_factor,rhs,check_finite=False)

    r10=invert_zero(mean*constant-b1)
    resolve_angle=eps!=0. and not averaged
    if resolve_angle:
        phase_system=L+1j*harmonic*V
        if phase_mixing_rate:
            phase_system-=phase_mixing_rate*harmonic**2*np.eye(len(p))
        phase_factor=lu_factor(phase_system,check_finite=False)
        r11=lu_solve(phase_factor,-eps*b1/2,check_finite=False)
    else:
        r11=np.zeros(len(p))

    first_zero=A1@(r10+eps*r11.real)
    c2=float(p@(w2_zero*b2+2*first_zero))
    r20=invert_zero(c2*constant+2*mean*r10-2*first_zero-w2_zero*b2)
    if resolve_angle:
        rhs=2*mean*r11-2*(A1@(r11+eps*r10/2))-eps*b2
        r21=lu_solve(phase_factor,rhs,check_finite=False)
    else:
        r21=np.zeros(len(p))
    c3=float(3*p@(A1@(r20+eps*r21.real)
                   +A2@(w2_zero*r10+2*eps*r11.real)))
    return dict(mean=-float(mean),noise=c2,third_cumulant=-c3)


@lru_cache(512)
def angular_readouts(temperature_a,temperature_b,relaxation_rate,flux_fraction,contrast,harmonic,drag):
    e=dict(temperature_a=temperature_a,temperature_b=temperature_b,relaxation_rate=relaxation_rate,
           flux_fraction=flux_fraction,angular_speed=0.,detector_contrast=contrast,detector_harmonic=harmonic)
    return angular_hermite(e,drag)

def predict_at(experiments,drag):
    result=[]
    for e in experiments:
        statistic=e.get('statistic','mean')
        if statistic not in ('mean','noise','third_cumulant'):
            raise ValueError('unknown calorimeter statistic')
        if e.get('detector_contrast',0.)!=0.:
            if e['angular_speed']!=0.:raise ValueError('Angularly gated preparations require zero flow target.')
            # Uniform initial angle stays uniform and independent of the
            # velocities at each time, so gating leaves the mean unchanged.
            if statistic=='mean':
                result.append(readout(*(float(e[f]) for f in FIELDS),statistic,float(drag)))
                continue
            values=angular_readouts(float(e['temperature_a']),float(e['temperature_b']),float(e['relaxation_rate']),
                float(e['flux_fraction']),float(e['detector_contrast']),int(e.get('detector_harmonic',1)),float(drag))
            result.append(values[statistic])
        else:result.append(readout(*(float(e[f]) for f in FIELDS),statistic,float(drag)))
    return np.asarray(result,dtype=float)


class Model:
    def __init__(self):self.drag=None

    def fit(self,records):
        """Fit the common drag by weighted least squares on calibration heat."""
        self.drag=fit_drag(records)
        return self

    def predict(self,experiments):
        if self.drag is None:
            raise ValueError('Fit the model before predicting.')
        return predict_at(experiments,self.drag)
