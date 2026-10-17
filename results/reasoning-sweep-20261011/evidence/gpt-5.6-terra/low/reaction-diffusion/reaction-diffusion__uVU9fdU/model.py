"""Bulk, electroneutral model for the electrolyte cell."""
import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse import diags

ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
LENGTH = 1.e-3

def equilibrium(totals):
    """Free A, free B, free C, P at ideal local equilibrium."""
    q = np.asarray(totals, dtype=float)
    if q.shape[-1] != 2: raise ValueError("totals must end in dimension 2")
    a, b = q[..., 0], q[..., 1]
    if np.any(a < 0) or np.any(b < 0): raise ValueError("negative concentration")
    lo = np.zeros(np.broadcast(a, b).shape); hi = np.minimum(a, b).copy()
    # The mass-action residual is monotone over the physically allowed extent.
    for _ in range(60):
        p = (lo + hi) / 2; af, bf = a - p, b - p
        f = p - ASSOCIATION * af * bf * (af + bf)**2
        lo = np.where(f < 0, p, lo); hi = np.where(f >= 0, p, hi)
    p = (lo + hi) / 2; af, bf = a-p, b-p
    return np.stack((af, bf, af+bf, p), axis=-1)

def _check_grid(t, x, initial):
    t = np.asarray(t, float); x = np.asarray(x, float); initial = np.asarray(initial, float)
    if t.ndim != 1 or not len(t) or np.any(t < 0) or np.any(np.diff(t) < 0):
        raise ValueError("t must be a nonempty increasing nonnegative array")
    if x.ndim != 1 or len(x) < 2 or initial.shape != (len(x), 2):
        raise ValueError("initial must have shape (len(x),2)")
    dx = np.diff(x)
    if not np.isfinite(x).all() or not np.allclose(dx, dx[0]): raise ValueError("x must be equally spaced")
    if not np.isfinite(initial).all() or np.any(initial < 0): raise ValueError("invalid initial concentrations")
    return t, x, initial

class TransportModel:
    def __init__(self): self.diffusivity = None

    def fit(self, data):
        x = np.asarray(data['x'], float); t = np.asarray(data['t'], float)
        initial = np.asarray(data['initial'], float); obs = np.asarray(data['concentration'], float)
        sigma = np.asarray(data['sigma'], float)
        if obs.shape != (len(initial),len(t),len(x),2) or sigma.shape != obs.shape:
            raise ValueError("calibration arrays have inconsistent shapes")
        if np.any(sigma <= 0) or not np.isfinite(sigma).all(): raise ValueError("invalid sigma")
        def loss(log_d):
            self.diffusivity = float(np.exp(log_d))
            pred = np.array([self.predict(t, x, start) for start in initial])
            return float(np.sum(((pred-obs)/sigma)**2))
        result = minimize_scalar(loss, bounds=(np.log(1e-13),np.log(1e-7)), method='bounded', options={'xatol':1e-12})
        if not result.success or not np.isfinite(result.fun): raise RuntimeError("could not fit diffusivity")
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self, t, x, initial):
        # Ambipolar A/C and B/C diffusivities: 4D/3, 8D/3.
        n = len(x)-1
        wave2 = 4/(x[1]-x[0])**2 * np.sin(np.arange(n+1)*np.pi/(2*n))**2
        coefficients = dct(initial, type=1, axis=0)
        rates = wave2[:,None]*self.diffusivity*np.array((4/3,8/3))
        return idct(np.exp(-t[:,None,None]*rates[None])*coefficients[None], type=1, axis=1)

    def predict(self, t, x, initial):
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit must be called before predict")
        t, x, initial = _check_grid(t,x,initial)
        if np.all(initial[:,0] == 0) or np.all(initial[:,1] == 0): return self._binary(t,x,initial)
        n, dx = len(x), x[1]-x[0]
        volume = np.full(n,dx); volume[[0,-1]] *= .5
        da,db,dc = self.diffusivity*np.array((1.,4.,2.))
        def rhs(_, flat):
            c = equilibrium(flat.reshape(n,2)); face = (c[:-1]+c[1:])/2; grad = np.diff(c,axis=0)/dx
            field = -(da*grad[:,0]+db*grad[:,1]-dc*grad[:,2])/(da*face[:,0]+db*face[:,1]+dc*face[:,2])
            ion_flux = -np.array((da,db))*(grad[:,:2]+face[:,:2]*field[:,None])
            flux = np.zeros((n+1,2))
            flux[1:-1] = ion_flux-self.diffusivity*NEUTRAL_MOBILITY*grad[:,3,None]
            return (-np.diff(flux,axis=0)/volume[:,None]).ravel()
        sparse = diags([np.ones(2*n-abs(k)) for k in range(-3,4)], range(-3,4),shape=(2*n,2*n),format='csr')
        sol = solve_ivp(rhs,(0,t[-1]),initial.ravel(),t_eval=t,method='BDF',rtol=2e-8,atol=1e-10,jac_sparsity=sparse)
        if not sol.success: raise RuntimeError(sol.message)
        return sol.y.T.reshape(len(t),n,2)

    def correlate(self, t, totals, harmonic=1):
        if self.diffusivity is None or self.diffusivity <= 0: raise RuntimeError("fit must be called before correlate")
        t = np.asarray(t,float); totals = np.asarray(totals,float)
        if t.ndim != 1 or np.any(t<0) or np.any(np.diff(t)<0): raise ValueError("invalid lag times")
        if totals.shape != (2,) or np.any(totals<=0) or not np.isfinite(totals).all(): raise ValueError("totals must be two positive values")
        if isinstance(harmonic,(bool,np.bool_)) or int(harmonic)!=harmonic or not 1 <= int(harmonic) <= 8: raise ValueError("harmonic must be an integer 1 through 8")
        susceptibility, mobility = fluctuation_matrices(totals,self.diffusivity)
        rate = (int(harmonic)*np.pi/LENGTH)**2 * mobility @ np.linalg.inv(susceptibility)
        # Specified detector normalization: independent total-number noise at zero lag.
        return np.array([expm(-lag*rate) @ np.diag(totals) for lag in t])

def fluctuation_matrices(totals, diffusivity):
    c = equilibrium(np.asarray(totals,float))
    charge = np.array((1.,1.,-1.,0.)); parts = np.array(((1.,0.,0.,1.),(0.,1.,0.,1.)))
    population = np.diag(c)
    constrained = population-np.outer(population@charge,population@charge)/(charge@population@charge)
    susceptibility = parts@constrained@parts.T
    particle_mobility = np.diag(diffusivity*np.array((1.,4.,2.,NEUTRAL_MOBILITY))*c)
    mq = particle_mobility@charge
    current_free = particle_mobility-np.outer(mq,mq)/(charge@particle_mobility@charge)
    return susceptibility, parts@current_free@parts.T
