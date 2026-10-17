import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
from scipy.sparse import diags

# Neutral four-ion association A+ + B+ + 2 C- <-> P.
ASSOCIATION=0.5
NEUTRAL_MOBILITY=0.15

def equilibrium(total):
    a,b=np.asarray(total).T
    bound=np.zeros_like(a)
    for _ in range(24):
        free_a=a-bound;free_b=b-bound;anion=free_a+free_b
        value=bound-ASSOCIATION*free_a*free_b*anion**2
        derivative=1+ASSOCIATION*(anion**3+4*free_a*free_b*anion)
        change=value/derivative
        bound=np.clip(bound-change,0.,np.minimum(a,b))
        if np.max(abs(change))<1e-13:break
    return np.column_stack((a-bound,b-bound,a+b-2*bound,bound))

class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self,data):
        """Fit the one (positive) diffusion scale by Gaussian least squares.

        The supplied calibration experiments are binary electrolytes.  In that
        case ``_binary`` is the exact semi-discrete, no-flux solution, making
        this both considerably faster and more reproducible than estimating D
        from a numerical time derivative.
        """
        required = ("x", "t", "initial", "concentration", "sigma")
        if any(name not in data for name in required):
            raise ValueError("data must contain x, t, initial, concentration, and sigma")
        x = np.asarray(data["x"], dtype=float)
        t = np.asarray(data["t"], dtype=float)
        initial = np.asarray(data["initial"], dtype=float)
        observed = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)
        if (x.ndim != 1 or t.ndim != 1 or len(x) < 2 or
                initial.ndim != 3 or initial.shape[1:] != (len(x), 2) or
                observed.shape != (initial.shape[0], len(t), len(x), 2) or
                sigma.shape != observed.shape or not np.all(sigma > 0)):
            raise ValueError("calibration arrays have incompatible shapes")

        def loss(log_d):
            self.diffusivity = float(np.exp(log_d))
            predicted = np.array([self.predict(t, x, state) for state in initial])
            residual = (predicted - observed) / sigma
            return float(np.sum(residual * residual))

        # This deliberately broad range covers ordinary aqueous and slow-gel
        # apparatuses while keeping the scalar search well conditioned.
        optimum = minimize_scalar(loss, bounds=(np.log(1e-14), np.log(1e-6)),
                                  method="bounded", options={"xatol": 1e-11})
        if not optimum.success or not np.isfinite(optimum.fun):
            raise RuntimeError("unable to fit diffusivity")
        self.diffusivity = float(np.exp(optimum.x))
        return self

    def _binary(self,t,x,initial):
        t,x,initial=np.asarray(t),np.asarray(x),np.asarray(initial)
        count=len(x)-1
        spatial_rate=4/(x[1]-x[0])**2*np.sin(np.arange(count+1)*np.pi/(2*count))**2
        coefficients=dct(initial,type=1,axis=0)
        rates=spatial_rate[:,None]*self.diffusivity*np.array([4/3,8/3])[None,:]
        return idct(np.exp(-t[:,None,None]*rates[None,:,:])*coefficients[None,:,:],type=1,axis=1)

    def predict(self,t,x,initial):
        t,x,initial=np.asarray(t),np.asarray(x),np.asarray(initial)
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit must be called before predict")
        if (t.ndim != 1 or x.ndim != 1 or initial.shape != (len(x), 2) or
                len(x) < 2 or np.any(np.diff(x) <= 0) or np.any(t < 0) or
                np.any(np.diff(t) < 0) or
                not np.allclose(np.diff(x), x[1]-x[0], rtol=1e-10, atol=1e-15)):
            raise ValueError("invalid time, mesh, or initial concentration array")
        if np.any(initial < 0):
            raise ValueError("initial concentrations must be nonnegative")
        if len(t) == 0:
            return np.empty((0, len(x), 2), dtype=float)
        if t[-1]==0:
            return np.broadcast_to(initial, (len(t),)+initial.shape).copy()
        if np.all(initial[:,0]==0) or np.all(initial[:,1]==0):
            return self._binary(t,x,initial)
        count=len(x);dx=x[1]-x[0]
        width=np.full(count,dx);width[[0,-1]]*=.5
        da,db,dc=self.diffusivity*np.array([1.,4.,2.])
        def derivative(time,flat):
            c=equilibrium(flat.reshape(count,2))
            face=(c[:-1]+c[1:])/2
            gradient=np.diff(c,axis=0)/dx
            potential_gradient=-(da*gradient[:,0]+db*gradient[:,1]-dc*gradient[:,2])/(da*face[:,0]+db*face[:,1]+dc*face[:,2])
            free_flux=-np.array([da,db])*(gradient[:,:2]+face[:,:2]*potential_gradient[:,None])
            bound_flux=-self.diffusivity*NEUTRAL_MOBILITY*gradient[:,3]
            flux=np.zeros((count+1,2))
            flux[1:-1]=free_flux+bound_flux[:,None]
            return (-np.diff(flux,axis=0)/width[:,None]).ravel()
        sparsity=diags([np.ones(2*count-abs(i)) for i in range(-3,4)],range(-3,4),format="csr")
        result=solve_ivp(derivative,(0,t[-1]),initial.ravel(),method="BDF",t_eval=t,rtol=1e-8,atol=1e-10,jac_sparsity=sparsity)
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T.reshape(len(t),count,2)

from scipy.linalg import expm

def fluctuation_matrices(totals,diffusivity):
    concentration=equilibrium(np.asarray(totals)[None,:])[0]
    charge=np.array([1.,1.,-1.,0.]);constituents=np.array([[1.,0.,0.,1.],[0.,1.,0.,1.]])
    population=np.diag(concentration)
    conditional=population-np.outer(concentration*charge,concentration*charge)/(charge@population@charge)
    susceptibility=constituents@conditional@constituents.T
    mobile=np.diag(diffusivity*np.array([1.,4.,2.,.15])*concentration)
    mobile_charge=mobile@charge
    constrained=mobile-np.outer(mobile_charge,mobile_charge)/(charge@mobile@charge)
    mobility=constituents@constrained@constituents.T
    return susceptibility,mobility

def equilibrium_correlations(t,totals,harmonic,diffusivity):
    covariance,mobility=fluctuation_matrices(totals,diffusivity)
    relaxation=(harmonic*np.pi/.001)**2*mobility@np.linalg.inv(covariance)
    initial_covariance=np.diag(totals)
    return np.array([expm(-float(time)*relaxation)@initial_covariance for time in t])

def correlate(self,t,totals,harmonic=1):
    if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
        raise RuntimeError("fit must be called before correlate")
    times = np.asarray(t, dtype=float)
    totals = np.asarray(totals, dtype=float)
    if (times.ndim != 1 or np.any(times < 0) or np.any(np.diff(times) < 0) or
            totals.shape != (2,) or np.any(totals < .2) or np.any(totals > 2.) or
            not isinstance(harmonic, (int, np.integer)) or not 1 <= harmonic <= 8):
        raise ValueError("invalid correlation time, totals, or harmonic")
    return equilibrium_correlations(times, totals, harmonic, self.diffusivity)

TransportModel.correlate=correlate
