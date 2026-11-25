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
    # At second order the contacts need not retain equal conditional
    # populations.  The density moment equation is
    #
    #     q P_2 = d_x M_{x,1} .
    #
    # Its component parallel to the stationary contact distribution is the
    # (irrelevant here) common positional-density correction: it cancels
    # exactly against B_m R_m.  Set that component to zero, while retaining
    # the conditional part, which is essential when bath temperatures or
    # contact drags differ.
    density_rhs=np.array([d@first[1,0][s] for s in range(count)])
    augmented=np.zeros((count+1,count+1))
    augmented[:count,:count]=q
    augmented[:count,count]=1.
    augmented[count,:count]=1.
    density_conditional=np.linalg.solve(
        augmented,np.vstack((density_rhs,np.zeros((1,points))))
    )[:count]
    density_correction={(0,0):density_conditional}
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
    """Calibrated predictor for the limiting excess calorimetry rate.

    ``friction`` is deliberately the only fitted parameter.  The quantities in
    a calibration record are noisy measurements of independent experiments, so
    fitting is a (one dimensional) weighted least-squares problem.
    """
    def __init__(self):
        self.friction=None

    def fit(self,records):
        """Estimate the common drag from calibration records and return ``self``.

        A small self-contained golden-section search is used instead of making
        the public model depend on a particular optimisation package.  Keeping
        the endpoints in the search also makes the documented, closed fitting
        range meaningful when a data set favours a boundary value.
        """
        records=list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        try:
            experiments=[record['input'] for record in records]
            observed=np.asarray([record['value'] for record in records],dtype=float)
            sigma=np.asarray([record['sigma'] for record in records],dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each record must contain input, value, and sigma") from exc
        if (not np.isfinite(observed).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError("calibration values must be finite and sigmas positive")

        def loss(friction):
            prediction=coefficients(experiments,float(friction))
            residual=(prediction-observed)/sigma
            return float(residual@residual)

        # The likelihood is smooth on the physical interval.  70 iterations
        # leaves an interval far below double precision at this scale.
        lo,hi=0.7,1.6
        ratio=(np.sqrt(5.0)-1.0)/2.0
        left=hi-ratio*(hi-lo)
        right=lo+ratio*(hi-lo)
        f_left,f_right=loss(left),loss(right)
        for _ in range(70):
            if f_left <= f_right:
                hi,right,f_right=right,left,f_left
                left=hi-ratio*(hi-lo)
                f_left=loss(left)
            else:
                lo,left,f_left=left,right,f_right
                right=lo+ratio*(hi-lo)
                f_right=loss(right)

        candidates=(0.7,1.6,lo,(lo+hi)/2.0,hi)
        self.friction=float(min(candidates,key=loss))
        return self

    def predict(self,experiments):
        if self.friction is None:
            raise RuntimeError("call fit(records) before predict(experiments)")
        result=coefficients(list(experiments),self.friction)
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite calorimetry prediction")
        return result
