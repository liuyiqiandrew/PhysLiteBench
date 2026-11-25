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
    # The next even term in the Kramers expansion is needed for a local
    # detector.  In particular, setting the order-two density to zero is OK
    # only after integrating an *unweighted* readout.  Solve for its contact
    # part first (q p2 = d_x u1), then determine its marginal part from the
    # next flux solvability condition.
    stationary=np.full(count,1./count)
    augmented=np.empty((count+1,count+1))
    augmented[:count,:count]=q
    augmented[:count,count]=stationary
    augmented[count,:count]=1.
    augmented[count,count]=0.
    rhs=(d@first[1,0].T).T
    # Roundoff in the leading stationary solve can leave a tiny component
    # outside range(q); remove it before applying the constrained inverse.
    rhs=rhs-stationary[:,None]*rhs.sum(axis=0,keepdims=True)
    particular=np.linalg.solve(augmented,np.vstack((rhs,np.zeros((1,points)))))[:count]

    def second_at(p2):
        return solve_moments(2,{(0,0):p2},temperatures,gx,gy,magnetic,q,
                             d,force,first,third)

    # First obtain the inhomogeneous flux, with zero marginal correction.
    second_base=second_at(particular)
    first_base=solve_moments(1,{},temperatures,gx,gy,magnetic,q,d,force,
                             {(0,0):particular},second_base)
    base_flux=first_base[1,0].sum(axis=0)

    # For a marginal h, p2_s=pi_s h.  Write the corresponding next flux as
    # a(x) h'(x)+b(x) h.  This avoids a large block solve while retaining the
    # full contact-resolved fast dynamics.
    p_unit=np.repeat(stationary[:,None],points,axis=1)
    second_unit=second_at(p_unit)
    pairs,a1=moment_operator(1,gx,gy,magnetic,q)
    inv_a1=np.linalg.inv(a1)
    i20={(p,n):i for i,(p,n) in enumerate(pairs)}
    # Sources are ordered by the degree-one basis [(1,0),(0,1)].
    c=np.zeros((count*2,points))
    cp=np.zeros_like(c)
    for s in range(count):
        c[2*s+i20[1,0]]=second_unit[2,0][s]
        c[2*s+i20[0,1]]=second_unit[1,1][s]
        cp[2*s+i20[1,0]]=d@second_unit[2,0][s]
        cp[2*s+i20[0,1]]=d@second_unit[1,1][s]
    fsrc=np.zeros_like(c)
    for s in range(count):
        fsrc[2*s+i20[1,0]]=force[0]*stationary[s]
        fsrc[2*s+i20[0,1]]=force[1]*stationary[s]
    # a multiplies h' and b multiplies h in the x component of u1^(3).
    response_derivative=inv_a1@c
    response_value=inv_a1@(cp-fsrc)
    a=response_derivative.reshape(count,2,points)[:,i20[1,0]].sum(axis=0)
    b=response_value.reshape(count,2,points)[:,i20[1,0]].sum(axis=0)
    marginal_operator=d@(a[:,None]*d + np.diag(b))
    marginal_rhs=-d@base_flux
    marginal_operator[-1]=dx
    marginal_rhs[-1]=0.
    marginal=np.linalg.solve(marginal_operator,marginal_rhs)
    density_correction={(0,0):particular+stationary[:,None]*marginal}
    second_correction=second_at(density_correction[0,0])

    # The uniform background has a 1/m switched-contact entropy throughput.
    # Its leading coefficient is local and proportional to the position
    # density; B_m R_m removes C times the order-two marginal density too.
    background_second=solve_moments(2,unit0,factors[:,None],gx,gy,magnetic,q)
    background_c=np.sum(gx*(background_second[2,0][:,0]/factors-1./count)
                        +gy*(background_second[0,2][:,0]/factors-1./count))
    local=np.sum(gx[:,None]*(second_correction[2,0]/temperatures-density_correction[0,0])
                 +gy[:,None]*(second_correction[0,2]/temperatures-density_correction[0,0]),axis=0)
    local-=background_c*marginal
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
        """Weighted least-squares estimate of the shared drag coefficient."""
        records=list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments=[r['input'] for r in records]
        observed=np.asarray([r['value'] for r in records],dtype=float)
        sigma=np.asarray([r['sigma'] for r in records],dtype=float)
        if (not np.isfinite(observed).all() or not np.isfinite(sigma).all()
                or np.any(sigma<=0)):
            raise ValueError("calibration values and sigmas must be finite, with sigma > 0")

        def objective(friction):
            residual=(coefficients(experiments,float(friction))-observed)/sigma
            return float(residual@residual)

        # A bounded golden-section search keeps the module dependent only on
        # NumPy and is ample for this one smooth, bounded parameter.
        lo,hi=.7,1.6
        phi=(5**.5-1)/2
        left=hi-phi*(hi-lo);right=lo+phi*(hi-lo)
        fleft=objective(left);fright=objective(right)
        for _ in range(42):
            if fleft<=fright:
                hi,right,fright=right,left,fleft
                left=hi-phi*(hi-lo);fleft=objective(left)
            else:
                lo,left,fleft=left,right,fright
                right=lo+phi*(hi-lo);fright=objective(right)
        self.friction=float((lo+hi)/2)
        return self
    def predict(self,experiments):return coefficients(experiments,self.friction)
