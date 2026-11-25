"""Stationary small-mass calorimetry of switched anisotropic thermal contacts."""
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

INPUT_FIELDS=('force_x','force_y','drag_ratio','magnetic','temperature','contrast','wavenumber','bath_contrast','switch_rate')

@lru_cache(None)
def spatial_grid(points):
    phase=np.arange(points)*2*np.pi/points
    modes=np.fft.fftfreq(points,1/points)
    derivative=np.fft.ifft(1j*modes[:,None]*np.fft.fft(np.eye(points),axis=0),axis=0).real
    return phase,derivative

def contact_parameters(e,friction):
    rate=e['switch_rate']
    if 'valve_cycle' not in e:
        labels=np.array([1.,-1.]);q=np.array([[-rate,rate],[rate,-rate]])
        drag_scale=np.ones(2)
    else:
        labels=np.array([-1.,0.,1.]);bias=e['valve_cycle'];q=np.zeros((3,3))
        for s in range(3):
            q[(s+1)%3,s]=rate*(1+bias)
            q[(s-1)%3,s]=rate*(1-bias)
            q[s,s]=-2*rate
        drag_scale=1+e.get('contact_drag_contrast',0.)*labels
    return 1+e['bath_contrast']*labels,friction*drag_scale,friction*drag_scale*e['drag_ratio'],q

def moment_operator(degree,gx,gy,magnetic,q):
    count=len(gx);basis=[(p,degree-p) for p in range(degree,-1,-1)];lookup={pair:i for i,pair in enumerate(basis)}
    a=np.kron(q,np.eye(degree+1))
    for s in range(count):
        for i,(p,n) in enumerate(basis):
            row=s*(degree+1)+i;a[row,row]-=p*gx[s]+n*gy[s]
            if p:a[row,s*(degree+1)+lookup[p-1,n+1]]+=p*magnetic
            if n:a[row,s*(degree+1)+lookup[p+1,n-1]]-=n*magnetic
    return basis,a

def solve_moments(degree,lower,temperature,gx,gy,magnetic,q,derivative=None,force=(0.,0.),prior=None,higher=None):
    count,points=temperature.shape;basis,a=moment_operator(degree,gx,gy,magnetic,q)
    source=np.zeros((count,degree+1,points))
    for s in range(count):
        for i,(p,n) in enumerate(basis):
            if p>=2:source[s,i]+=p*(p-1)*gx[s]*temperature[s]*lower[p-2,n][s]
            if n>=2:source[s,i]+=n*(n-1)*gy[s]*temperature[s]*lower[p,n-2][s]
            if prior is not None:
                if p:source[s,i]+=p*force[0]*prior[p-1,n][s]
                if n:source[s,i]+=n*force[1]*prior[p,n-1][s]
            if higher is not None:source[s,i]-=derivative@higher[p+1,n][s]
    value=np.linalg.solve(a,-source.reshape(count*(degree+1),points)).reshape(count,degree+1,points)
    return {pair:value[:,i] for i,pair in enumerate(basis)}

def limiting_state(e,friction,points=129):
    phase,d=spatial_grid(points);d=d*e['wavenumber'];dx=2*np.pi/points
    factors,gx,gy,q=contact_parameters(e,friction);count=len(gx)
    force=(e['force_x'],e['force_y']);magnetic=e['magnetic']
    local_t=e['temperature']*(1+e['contrast']*np.cos(phase))
    temperatures=factors[:,None]*local_t
    # Fast velocity moments per unit positional density and base temperature.
    unit0={(0,0):np.ones((count,1))/count}
    unit2=solve_moments(2,unit0,factors[:,None],gx,gy,magnetic,q)
    _,a1=moment_operator(1,gx,gy,magnetic,q)
    force_source=np.tile(np.array(force)/count,count)
    diff_source=np.column_stack((unit2[2,0][:,0],unit2[1,1][:,0])).ravel()
    velocity=-np.linalg.solve(a1,force_source).reshape(count,2).sum(axis=0)
    diffusion=-np.linalg.solve(a1,diff_source).reshape(count,2).sum(axis=0)
    operator=-d@(velocity[0]*np.eye(points)-diffusion[0]*d*local_t[None,:])
    operator[-1]=dx;rhs=np.zeros(points);rhs[-1]=1.
    density=np.linalg.solve(operator,rhs)
    zeroth={(0,0):np.tile(density/count,(count,1))}
    second=solve_moments(2,zeroth,temperatures,gx,gy,magnetic,q)
    fourth=solve_moments(4,second,temperatures,gx,gy,magnetic,q)
    first=solve_moments(1,{},temperatures,gx,gy,magnetic,q,d,force,zeroth,second)
    third=solve_moments(3,first,temperatures,gx,gy,magnetic,q,d,force,second,fourth)
    # Use the leading equal conditional contact populations.
    density_correction={(0,0):np.zeros((count,points))}
    second_correction=solve_moments(2,density_correction,temperatures,gx,gy,magnetic,q,d,force,first,third)
    local=np.sum(gx[:,None]*(second_correction[2,0]/temperatures-density_correction[0,0])+gy[:,None]*(second_correction[0,2]/temperatures-density_correction[0,0]),axis=0)
    return dict(value=float(dx*local.sum()),local=local,phase=phase,density=density,diffusion=diffusion,velocity=velocity,dx=dx,first=first,second=second,density_correction=density_correction,second_correction=second_correction,temperatures=temperatures)

@lru_cache(4096)
def entropy_rate(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate,friction,detector_phase=None,valve_cycle=None,contact_drag_contrast=0.):
    if bath_contrast==0. and magnetic==0. and valve_cycle is None and friction!=1.:
        return entropy_rate(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate,1.,detector_phase)/friction
    e=dict(zip(INPUT_FIELDS,(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate)))
    if valve_cycle is not None:e.update(valve_cycle=valve_cycle,contact_drag_contrast=contact_drag_contrast)
    state=limiting_state(e,friction)
    weight=1. if detector_phase is None else (1+np.cos(state['phase']-detector_phase))/2
    return float(state['dx']*np.sum(weight*state['local']))

def coefficients(experiments,friction=1.):
    return np.array([entropy_rate(*(e[f] for f in INPUT_FIELDS),float(friction),e.get('detector_phase'),e.get('valve_cycle'),e.get('contact_drag_contrast',0.)) for e in experiments])

class Model:
    def __init__(self):self.friction=None
    def fit(self,records):
        """Fit the (single) drag coefficient by weighted least squares.

        All supplied calibration runs use the two-contact, zero-field setup.
        In that setup the limiting rate is exactly proportional to 1/friction,
        so fitting its reciprocal is a one-parameter linear regression.  This
        also makes the fit much less sensitive to the different uncertainties
        in the calibration file.
        """
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments=[r['input'] for r in records]
        observed=np.asarray([r['value'] for r in records],dtype=float)
        sigma=np.asarray([r['sigma'] for r in records],dtype=float)
        if (not np.isfinite(observed).all() or not np.isfinite(sigma).all()
                or np.any(sigma<=0)):
            raise ValueError("calibration values and sigmas must be finite, with positive sigma")
        unit=coefficients(experiments,1.)
        inverse=float(np.sum(unit*observed/sigma**2)/np.sum(unit**2/sigma**2))
        if not np.isfinite(inverse) or inverse<=0:
            raise ValueError("calibration data do not identify a positive friction")
        self.friction=float(np.clip(1./inverse,.7,1.6))
        return self

    def predict(self,experiments):
        if self.friction is None:
            raise RuntimeError("call fit before predict")
        return np.asarray([_excess_rate(e,self.friction) for e in experiments],dtype=float)


# The calculation below intentionally works at small, nonzero mass before
# taking the limit.  This matters when bath_contrast is nonzero: the two raw
# calorimeter rates contain a 1/m switching heat leak.  It cancels only after
# multiplying the background rate by the *measured* foreground residence.

_MOMENT_POINTS=33
_MOMENT_ORDER=4

def _finite_mass_state(e,friction,mass):
    """Return local entropy rate and positional density at finite mass.

    We solve the stationary Kramers equation in scaled velocities u=sqrt(m)v.
    Its polynomial moment hierarchy is sparse; order four supplies the
    finite-mass error needed for the linear Richardson limit below.  Fourier
    collocation in x is spectrally accurate for the smooth periodic profiles.
    """
    points=_MOMENT_POINTS
    order=_MOMENT_ORDER
    phase,derivative=spatial_grid(points)
    derivative=derivative*e['wavenumber']
    dx=2*np.pi/points
    factors,gx,gy,q=contact_parameters(e,friction)
    contacts=len(gx)
    temperature=(e['temperature']*(1+e['contrast']*np.cos(phase))[None,:]
                 *factors[:,None])
    epsilon=np.sqrt(mass)

    # offsets for all homogeneous velocity-moment degrees 0,...,order
    bases=[]
    offsets=[]
    size=0
    for degree in range(order+1):
        basis=[(p,degree-p) for p in range(degree,-1,-1)]
        bases.append(basis)
        offsets.append(size)
        size+=contacts*(degree+1)*points

    identity=sparse.eye(points,format='csr')
    blocks=[]
    def add(row_offset,column_offset,block):
        blocks.append((row_offset,column_offset,block))

    for degree,basis in enumerate(bases):
        # Fast Ornstein--Uhlenbeck/contact part for moments of this degree.
        _,fast=moment_operator(degree,gx,gy,e['magnetic'],q)
        add(offsets[degree],offsets[degree],
            sparse.kron(sparse.csr_matrix(fast),identity,format='csr'))
        lower_lookup={pair:i for i,pair in enumerate(bases[degree-1])} if degree else {}
        lower2_lookup={pair:i for i,pair in enumerate(bases[degree-2])} if degree>=2 else {}
        higher_lookup={pair:i for i,pair in enumerate(bases[degree+1])} if degree<order else {}
        for s in range(contacts):
            for i,(px,py) in enumerate(basis):
                row=offsets[degree]+(s*(degree+1)+i)*points
                if px>=2:
                    col=offsets[degree-2]+(s*(degree-1)+lower2_lookup[px-2,py])*points
                    add(row,col,sparse.diags(px*(px-1)*gx[s]*temperature[s]))
                if py>=2:
                    col=offsets[degree-2]+(s*(degree-1)+lower2_lookup[px,py-2])*points
                    add(row,col,sparse.diags(py*(py-1)*gy[s]*temperature[s]))
                if px:
                    col=offsets[degree-1]+(s*degree+lower_lookup[px-1,py])*points
                    add(row,col,epsilon*px*e['force_x']*identity)
                if py:
                    col=offsets[degree-1]+(s*degree+lower_lookup[px,py-1])*points
                    add(row,col,epsilon*py*e['force_y']*identity)
                if degree<order:
                    col=offsets[degree+1]+(s*(degree+2)+higher_lookup[px+1,py])*points
                    add(row,col,-epsilon*sparse.csr_matrix(derivative))

    rows=[]; columns=[]; values=[]
    for row_offset,column_offset,block in blocks:
        coo=block.tocoo()
        rows.extend((coo.row+row_offset).tolist())
        columns.extend((coo.col+column_offset).tolist())
        values.extend(coo.data.tolist())
    system=sparse.coo_matrix((values,(rows,columns)),shape=(size,size)).tolil()

    # The sole null mode is total probability.  Replace one redundant
    # stationarity equation by the normalization of the x marginal.
    system[0,:]=0.
    for s in range(contacts):
        system[0,offsets[0]+s*points:offsets[0]+(s+1)*points]=dx
    rhs=np.zeros(size)
    rhs[0]=1.
    solution=spsolve(system.tocsc(),rhs)
    density=solution[offsets[0]:offsets[1]].reshape(contacts,points)
    second=solution[offsets[2]:offsets[3]].reshape(contacts,3,points)

    # E[dS_bath]/dt = gamma ( <u_i^2>/T - rho ) / m.
    local=np.sum(gx[:,None]*(second[:,0]/temperature-density)
                 +gy[:,None]*(second[:,2]/temperature-density),axis=0)/mass
    return phase,dx,density,local

def _finite_excess(e,friction,mass):
    phase,dx,density,foreground_local=_finite_mass_state(e,friction,mass)
    if 'detector_phase' in e:
        weight=(1+np.cos(phase-e['detector_phase']))/2
    else:
        weight=np.ones_like(phase)
    weighted_foreground=dx*np.sum(weight*foreground_local)
    residence=dx*np.sum(weight*density)

    background=dict(e)
    background.update(force_x=0.,force_y=0.,contrast=0.)
    _,background_dx,_,background_local=_finite_mass_state(background,friction,mass)
    background_rate=background_dx*np.sum(background_local)
    return float(weighted_foreground-background_rate*residence)

@lru_cache(4096)
def _excess_rate_cached(force_x,force_y,drag_ratio,magnetic,temperature,contrast,
                        wavenumber,bath_contrast,switch_rate,friction,
                        detector_phase=None,valve_cycle=None,contact_drag_contrast=0.):
    e=dict(zip(INPUT_FIELDS,(force_x,force_y,drag_ratio,magnetic,temperature,
                             contrast,wavenumber,bath_contrast,switch_rate)))
    if detector_phase is not None:
        e['detector_phase']=detector_phase
    if valve_cycle is not None:
        e['valve_cycle']=valve_cycle
        e['contact_drag_contrast']=contact_drag_contrast
    # The expansion is even in sqrt(m), hence the subtracted observable is
    # L+c*m+O(m^2).  Two small masses remove the leading finite-mass term.
    coarse=_finite_excess(e,friction,.002)
    fine=_finite_excess(e,friction,.001)
    return 2*fine-coarse

def _excess_rate(e,friction):
    return _excess_rate_cached(*(e[name] for name in INPUT_FIELDS),float(friction),
        e.get('detector_phase'),e.get('valve_cycle'),e.get('contact_drag_contrast',0.))
