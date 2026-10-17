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
        labels=np.array([1.,-1.]);q=np.array([[-rate,rate],[rate,-rate]],dtype=float)
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

def contact_correction(q,derivative,current):
    """Return the order-m contact-state correction to the density.

    The zeroth velocity moment of the stationary Fokker--Planck equation is

        q P_1 = d_x H_{1,x},

    where ``H_1`` is the leading scaled first velocity moment.  The right
    hand side has zero sum by the leading stationary flux equation.  The
    null-space (a common spatial density correction) is deliberately fixed to
    zero: that part cancels against the measured-residence background term.
    """
    count,points=current.shape
    matrix=q.copy()
    matrix[-1,:]=1./count
    rhs=(derivative@current.T).T
    rhs[-1,:]=0.
    return np.linalg.solve(matrix,rhs)

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
    # The total order-m density correction cancels against B_m*R_m, but its
    # contact-resolved part is physical when the reservoirs have different
    # temperatures.  Recover it from the zeroth-moment stationarity equation.
    density_correction={(0,0):contact_correction(q,d,first[1,0])}
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
        """Fit the common friction from the (weighted) calibration records.

        The calibration values are independent Gaussian measurements, so the
        maximum-likelihood estimate is the bounded weighted least-squares
        minimizer.  Keeping the one-dimensional optimization here (rather
        than fitting a separate coefficient for every record) is important:
        the switch rate and the spatial profile are part of each experiment,
        while ``friction`` is common to all of them.
        """
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments=[]
        values=[]
        sigmas=[]
        for record in records:
            if not isinstance(record, dict) or 'input' not in record:
                raise ValueError("each calibration record must contain input, value and sigma")
            try:
                value=float(record['value'])
                sigma=float(record['sigma'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("each calibration record must contain input, value and sigma") from exc
            if not np.isfinite(value) or not np.isfinite(sigma) or sigma<=0:
                raise ValueError("calibration values and uncertainties must be finite, with sigma > 0")
            experiments.append(record['input'])
            values.append(value)
            sigmas.append(sigma)

        values=np.asarray(values,dtype=float)
        weights=1./np.asarray(sigmas,dtype=float)**2

        def objective(friction):
            prediction=coefficients(experiments,float(friction))
            residual=prediction-values
            return float(np.dot(weights*residual,residual))

        # A bounded scalar search is sufficient and avoids making scipy part
        # of the model's public/runtime dependencies.  The calibration
        # objective is smooth; a modest grid also makes the result robust if
        # a future calibration set has a shallow minimum.
        lo,hi=.7,1.6
        grid=np.linspace(lo,hi,19)
        scores=np.array([objective(f) for f in grid])
        best=int(np.argmin(scores))
        left=grid[max(0,best-1)]
        right=grid[min(len(grid)-1,best+1)]
        if left==right:
            self.friction=float(left)
            return self
        golden=(np.sqrt(5.)-1.)/2.
        x1=right-golden*(right-left)
        x2=left+golden*(right-left)
        f1,f2=objective(x1),objective(x2)
        for _ in range(60):
            if f1>f2:
                left=x1
                x1=x2; f1=f2
                x2=left+golden*(right-left)
                f2=objective(x2)
            else:
                right=x2
                x2=x1; f2=f1
                x1=right-golden*(right-left)
                f1=objective(x1)
        self.friction=float((left+right)/2.)
        return self

    def predict(self,experiments):
        if self.friction is None:
            raise RuntimeError("Model.fit must be called before predict")
        result=coefficients(experiments,float(self.friction))
        # Keep the documented output contract even for an empty input.
        return np.asarray(result,dtype=float).reshape(-1)
