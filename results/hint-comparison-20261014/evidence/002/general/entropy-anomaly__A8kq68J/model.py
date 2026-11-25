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

def contact_density_correction(q, derivative, first_x):
    """The order-m contact population correction with zero total density.

    The contact generator is singular: its missing component is precisely the
    (as yet undetermined) correction to the positional density.  It cancels
    against the measured residence in the requested background subtraction,
    so fixing that component to zero is both convenient and well conditioned.
    """
    count, points=first_x.shape
    rhs=np.asarray([derivative@first_x[s] for s in range(count)])
    augmented=q.copy()
    augmented[-1]=1.
    rhs=rhs.copy()
    rhs[-1]=0.
    return np.linalg.solve(augmented,rhs)

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
    # At the next order the contact populations are no longer equal.  This
    # term is immaterial for the calibration (whose contacts are identical),
    # but is needed when the baths or their drags differ.
    density_correction={(0,0):contact_density_correction(q,d,first[1,0])}
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
    measured=float(state['dx']*np.sum(weight*state['local']))

    # The leading switching heat is O(1/m).  It is proportional to the
    # positional density and therefore cancels exactly against B_m R_m.  The
    # finite part still requires the matched, uniform and force-free
    # background specified in the problem statement.
    background=e.copy()
    background.update(force_x=0.,force_y=0.,contrast=0.)
    baseline=limiting_state(background,friction)['value']
    residence=float(state['dx']*np.sum(weight*state['density']))
    return measured-baseline*residence

def coefficients(experiments,friction=1.):
    return np.array([entropy_rate(*(e[f] for f in INPUT_FIELDS),float(friction),e.get('detector_phase'),e.get('valve_cycle'),e.get('contact_drag_contrast',0.)) for e in experiments])

class Model:
    def __init__(self):self.friction=None
    def fit(self,records):
        if len(records)==0:
            raise ValueError("at least one calibration record is required")
        experiments=[r['input'] for r in records]
        observed=np.asarray([r['value'] for r in records],dtype=float)
        sigma=np.asarray([r['sigma'] for r in records],dtype=float)
        if not (np.isfinite(observed).all() and np.isfinite(sigma).all() and np.all(sigma>0)):
            raise ValueError("calibration values and sigmas must be finite, with positive sigmas")

        def loss(friction):
            residual=(coefficients(experiments,friction)-observed)/sigma
            return float(residual@residual)

        # A bounded golden-section search needs no optional dependency and is
        # stable for the one shared physical parameter.
        left,right=0.7,1.6
        ratio=(np.sqrt(5.)-1.)/2.
        x1=right-ratio*(right-left); x2=left+ratio*(right-left)
        y1,y2=loss(x1),loss(x2)
        for _ in range(48):
            if y1<=y2:
                right,x2,y2=x2,x1,y1
                x1=right-ratio*(right-left); y1=loss(x1)
            else:
                left,x1,y1=x1,x2,y2
                x2=left+ratio*(right-left); y2=loss(x2)
        candidates=(left,right,(left+right)/2,0.7,1.6)
        self.friction=float(min(candidates,key=loss))
        return self
    def predict(self,experiments):return coefficients(experiments,self.friction)
