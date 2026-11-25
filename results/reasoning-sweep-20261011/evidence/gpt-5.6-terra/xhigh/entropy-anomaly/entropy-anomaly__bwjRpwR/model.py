"""Stationary small-mass calorimetry of switched anisotropic thermal contacts."""
from functools import lru_cache
import numpy as np

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

def contact_density_correction(q,first,derivative):
    """Return the order-m, zero-total contact-population correction.

    The leading position density has equal (or, more generally, stationary)
    contact populations.  At the next even order the individual populations
    need not be stationary: their source is the divergence of the order-sqrt(m)
    velocity moment.  The total density at this order is deliberately fixed to
    zero here.  Its only effect on the requested observable is cancelled by
    the measured-residence background subtraction.
    """
    source=(derivative@first[1,0].T).T
    # The leading stationary equation makes sum(source) zero.  Remove the
    # round-off-sized violation before solving a singular generator equation.
    source=source-source.mean(axis=0,keepdims=True)
    system=q.copy()
    system[-1,:]=1.
    rhs=source.copy()
    rhs[-1]=0.
    return np.linalg.solve(system,rhs)

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
    # This is the non-stationary part of the order-m contact distribution.
    # Its total is set to zero; the undetermined total positional correction
    # cancels exactly against B_m R_m in the definition of the excess rate.
    density_correction={(0,0):contact_density_correction(q,first,d)}
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
        """Estimate the one shared drag coefficient by weighted least squares."""
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments=[record['input'] for record in records]
        observed=np.asarray([record['value'] for record in records],dtype=float)
        sigma=np.asarray([record['sigma'] for record in records],dtype=float)
        if (not np.isfinite(observed).all() or not np.isfinite(sigma).all()
                or np.any(sigma<=0)):
            raise ValueError("calibration values and positive sigmas must be finite")
        weight=1./sigma**2

        # The documented calibration protocol has identical contacts and no
        # magnetic field.  In that case a common rescaling of both drags only
        # rescales the excess rate, so the weighted fit has a closed form.
        simple=all(
            e.get('bath_contrast',0.)==0. and e.get('magnetic',0.)==0.
            and 'valve_cycle' not in e for e in experiments
        )
        if simple:
            unit=coefficients(experiments,1.)
            denominator=float(np.dot(weight*unit,unit))
            if denominator>0 and np.isfinite(denominator):
                inverse_friction=float(np.dot(weight*unit,observed)/denominator)
                self.friction=float(np.clip(1./inverse_friction,.7,1.6)) if inverse_friction>0 else 1.6
                return self

        def loss(friction):
            residual=(coefficients(experiments,float(friction))-observed)/sigma
            return float(residual@residual)

        # A bounded golden-section search avoids making SciPy part of the API.
        lower,upper=.7,1.6
        ratio=(np.sqrt(5.)-1.)/2.
        left=upper-ratio*(upper-lower)
        right=lower+ratio*(upper-lower)
        left_loss,right_loss=loss(left),loss(right)
        for _ in range(36):
            if left_loss<=right_loss:
                upper,right,right_loss=right,left,left_loss
                left=upper-ratio*(upper-lower)
                left_loss=loss(left)
            else:
                lower,left,left_loss=left,right,right_loss
                right=lower+ratio*(upper-lower)
                right_loss=loss(right)
        self.friction=float((lower+upper)/2.)
        return self

    def predict(self,experiments):
        if self.friction is None:
            raise RuntimeError("fit must be called before predict")
        return coefficients(experiments,float(self.friction))
