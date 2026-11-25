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

def _forward_poisson(q, rhs):
    """Solve q z = rhs with the zero-sum gauge, column-vector convention."""
    count, points = rhs.shape
    a = np.array(q, copy=True)
    a[-1] = 1.
    b = np.array(rhs, copy=True)
    b[-1] = 0.
    return np.linalg.solve(a, b)


def limiting_state(e,friction,points=129):
    phase,d=spatial_grid(points);d=d*e['wavenumber'];dx=2*np.pi/points
    factors,gx,gy,q=contact_parameters(e,friction);count=len(gx)
    pi=np.ones(count)/count
    force=(e['force_x'],e['force_y']);magnetic=e['magnetic']
    local_t=e['temperature']*(1+e['contrast']*np.cos(phase))
    temperatures=factors[:,None]*local_t

    # The leading velocity cell problem, for unit positional density.  The
    # contact generator is retained in these solves; it cannot in general be
    # replaced by the stationary contact probabilities.
    unit0={(0,0):pi[:,None]}
    unit2=solve_moments(2,unit0,factors[:,None],gx,gy,magnetic,q)
    _,a1=moment_operator(1,gx,gy,magnetic,q)
    force_source=(pi[:,None]*np.asarray(force)).ravel()
    diff_source=np.column_stack((unit2[2,0][:,0],unit2[1,1][:,0])).ravel()
    velocity=-np.linalg.solve(a1,force_source).reshape(count,2).sum(axis=0)
    diffusion=-np.linalg.solve(a1,diff_source).reshape(count,2).sum(axis=0)
    operator=-d@(velocity[0]*np.eye(points)-diffusion[0]*d*local_t[None,:])
    operator[-1]=dx;rhs=np.zeros(points);rhs[-1]=1.
    density=np.linalg.solve(operator,rhs)

    zeroth={(0,0):pi[:,None]*density[None,:]}
    second=solve_moments(2,zeroth,temperatures,gx,gy,magnetic,q)
    fourth=solve_moments(4,second,temperatures,gx,gy,magnetic,q)
    first=solve_moments(1,{},temperatures,gx,gy,magnetic,q,d,force,zeroth,second)
    third=solve_moments(3,first,temperatures,gx,gy,magnetic,q,d,force,second,fourth)

    # At first order the forward equation is
    #       q rho_1 = d_x j_0 .
    # Solve this conditional-population Poisson equation with the complete
    # forward generator, then determine its null-space component from the
    # next-order zero-flux condition.
    rhs_population=(d@first[1,0].T).T
    density_correction=_forward_poisson(q,rhs_population)
    second_part=solve_moments(2,{(0,0):density_correction},temperatures,gx,gy,magnetic,q,d,force,first,third)
    first_part=solve_moments(1,{},temperatures,gx,gy,magnetic,q,d,force,{(0,0):density_correction},second_part)
    j1_part=first_part
    j1_flux=np.sum(j1_part[1,0],axis=0)

    # A null-space contribution is pi_s*h(x).  Its flux uses precisely the
    # same effective mobility and diffusion obtained from the velocity cell
    # equations, including state-dependent drag.
    h_operator=-d@(velocity[0]*np.eye(points)-diffusion[0]*d*local_t[None,:])
    h_rhs=d@j1_flux
    h_operator[-1]=dx
    h_rhs[-1]=0.
    h=np.linalg.solve(h_operator,h_rhs)
    density_correction={(0,0):density_correction+pi[:,None]*h[None,:]}

    second_correction=solve_moments(2,density_correction,temperatures,gx,gy,magnetic,q,d,force,first,third)
    local=np.sum(gx[:,None]*(second_correction[2,0]/temperatures-density_correction[0,0])+gy[:,None]*(second_correction[0,2]/temperatures-density_correction[0,0]),axis=0)

    # The O(m^-1) entropy term is C*rho(x).  The matched uniform background
    # has the same C; multiplying its divergent rate by the measured
    # residence leaves the finite correction -C*R_1.
    base_c=np.sum(gx*(unit2[2,0][:,0]/factors-pi)+gy*(unit2[0,2][:,0]/factors-pi))
    weight=1. if e.get('detector_phase') is None else (1+np.cos(phase-e['detector_phase']))/2
    residence_correction=dx*np.sum(weight*h)
    value=dx*np.sum(weight*local)-base_c*residence_correction
    return dict(value=float(value),local=local,phase=phase,density=density,diffusion=diffusion,velocity=velocity,dx=dx,first=first,second=second,density_correction=density_correction,second_correction=second_correction,temperatures=temperatures,leading_background=float(base_c),residence_correction=float(residence_correction))

@lru_cache(4096)
def entropy_rate(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate,friction,detector_phase=None,valve_cycle=None,contact_drag_contrast=0.):
    e=dict(zip(INPUT_FIELDS,(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate)))
    if valve_cycle is not None:e.update(valve_cycle=valve_cycle,contact_drag_contrast=contact_drag_contrast)
    state=limiting_state(e,friction)
    return state['value']

def coefficients(experiments,friction=1.):
    return np.array([entropy_rate(*(e[f] for f in INPUT_FIELDS),float(friction),e.get('detector_phase'),e.get('valve_cycle'),e.get('contact_drag_contrast',0.)) for e in experiments])

class Model:
    def __init__(self):self.friction=None
    def fit(self,records):
        if not records:
            raise ValueError('calibration records must not be empty')

        inputs=[r['input'] for r in records]
        observed=np.asarray([r['value'] for r in records],float)
        sigma=np.asarray([r.get('sigma',1.) for r in records],float)
        if not np.isfinite(observed).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):
            raise ValueError('calibration values and uncertainties must be finite')

        def objective(friction):
            residual=(coefficients(inputs,friction)-observed)/sigma
            return float(residual@residual)

        # A bounded golden-section search is sufficient for the one shared
        # calibration parameter and avoids making scipy part of the model's
        # runtime requirements.
        left,right=.7,1.6
        phi=(1+np.sqrt(5.))/2
        x1=right-(right-left)/phi
        x2=left+(right-left)/phi
        f1,f2=objective(x1),objective(x2)
        for _ in range(32):
            if f1>f2:
                left=x1;x1=x2;f1=f2;x2=left+(right-left)/phi;f2=objective(x2)
            else:
                right=x2;x2=x1;f2=f1;x1=right-(right-left)/phi;f1=objective(x1)
        candidates=[(.7,objective(.7)),(1.6,objective(1.6)),(x1,f1),(x2,f2)]
        self.friction=float(min(candidates,key=lambda item:item[1])[0])
        return self
    def predict(self,experiments):return coefficients(experiments,self.friction)
