import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.linalg import expm

# Neutral four-ion association A+ + B+ + 2 C- <-> P.
ASSOCIATION=0.5
NEUTRAL_MOBILITY=0.15

def equilibrium(total):
    """Return (free A, free B, free C, bound) at local equilibrium.

    The only unknown in the local equilibrium problem is the concentration
    of P.  Its mass-action residual is monotone on [0, min(A, B)], so a
    vectorized bisection is both simple and considerably safer than an
    unconstrained Newton iteration near a vanishing concentration.
    """
    total = np.asarray(total, dtype=float)
    if total.ndim == 0 or total.shape[-1] != 2:
        raise ValueError("total must have a final dimension of length 2")

    a = total[..., 0]
    b = total[..., 1]
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("total concentrations must be nonnegative")

    lower = np.zeros_like(a)
    upper = np.minimum(a, b)
    for _ in range(60):
        bound = (lower + upper) * 0.5
        free_a = a - bound
        free_b = b - bound
        free_c = free_a + free_b
        residual = bound - ASSOCIATION * free_a * free_b * free_c**2
        lower = np.where(residual < 0, bound, lower)
        upper = np.where(residual < 0, upper, bound)
    bound = (lower + upper) * 0.5
    return np.stack((a - bound, b - bound, a + b - 2 * bound, bound), axis=-1)

class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self,data):
        """Fit the common diffusion scale from the single-cation records."""
        initial = np.asarray(data["initial"], dtype=float)
        observed = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)
        t = np.asarray(data["t"], dtype=float)
        x = np.asarray(data["x"], dtype=float)

        if initial.ndim != 3 or initial.shape[-1] != 2:
            raise ValueError("initial must have shape (preparation, x, 2)")
        if observed.shape[:2] != (initial.shape[0], t.size):
            raise ValueError("concentration has incompatible dimensions")
        if observed.shape != sigma.shape:
            raise ValueError("concentration and sigma must have the same shape")
        if np.any(sigma <= 0):
            raise ValueError("sigma must be positive")

        # The calibration preparations have only A or only B.  In that case
        # electroneutrality gives an effective diffusivity of 4D/3 and 8D/3,
        # respectively, so the solution is a cosine series.
        def objective(log_diffusivity):
            self.diffusivity = float(np.exp(log_diffusivity))
            prediction = np.array([
                self._binary(t, x, preparation) for preparation in initial
            ])
            residual = (prediction - observed) / sigma
            return float(np.sum(residual * residual))

        result = minimize_scalar(
            objective,
            bounds=(np.log(1e-12), np.log(1e-6)),
            method="bounded",
            options={"xatol": 1e-11},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError("unable to fit the diffusion scale")
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self,t,x,initial):
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if self.diffusivity is None:
            raise RuntimeError("fit the model before predicting")
        if t.ndim != 1 or x.ndim != 1 or initial.shape != (x.size, 2):
            raise ValueError("incompatible t, x, and initial dimensions")
        if x.size < 2 or not np.all(np.diff(x) > 0):
            raise ValueError("x must be an increasing mesh with at least two points")
        if np.any(np.diff(t) < 0) or np.any(t < 0):
            raise ValueError("t must be increasing and nonnegative")

        count = x.size - 1
        dx = x[1] - x[0]
        if not np.allclose(np.diff(x), dx, rtol=1e-10, atol=1e-15):
            raise ValueError("x must be equally spaced")
        spatial_rate = 4 / dx**2 * np.sin(
            np.arange(count + 1) * np.pi / (2 * count)
        )**2
        coefficients = dct(initial, type=1, axis=0)
        rates = spatial_rate[:, None] * self.diffusivity * np.array([4 / 3, 8 / 3])[None, :]
        modes = np.exp(-t[:, None, None] * rates[None, :, :])
        return idct(modes * coefficients[None, :, :], type=1, axis=1)

    def predict(self,t,x,initial):
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if self.diffusivity is None:
            raise RuntimeError("fit the model before predicting")
        if t.ndim != 1 or x.ndim != 1:
            raise ValueError("t and x must be one-dimensional arrays")
        if initial.shape != (x.size, 2):
            raise ValueError("initial must have shape (len(x), 2)")
        if np.any(np.diff(t) < 0) or np.any(t < 0):
            raise ValueError("t must be increasing and nonnegative")
        if t.size == 0:
            return np.empty((0, x.size, 2), dtype=float)
        if t[-1] == 0:
            return np.broadcast_to(initial, (t.size,) + initial.shape).copy()
        if np.all(initial[:,0]==0) or np.all(initial[:,1]==0):
            return self._binary(t,x,initial)
        if x.size < 2 or not np.all(np.diff(x) > 0):
            raise ValueError("x must be an increasing mesh with at least two points")
        dx = x[1] - x[0]
        if not np.allclose(np.diff(x), dx, rtol=1e-10, atol=1e-15):
            raise ValueError("x must be equally spaced")
        count = x.size
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

    def correlate(self, t, totals, harmonic=1):
        if self.diffusivity is None:
            raise RuntimeError("fit the model before calculating correlations")
        return equilibrium_correlations(t, totals, harmonic, self.diffusivity)

def fluctuation_matrices(totals,diffusivity):
    totals = np.asarray(totals, dtype=float)
    if totals.shape != (2,) or np.any(totals <= 0):
        raise ValueError("totals must contain two positive concentrations")
    if not np.isfinite(diffusivity) or diffusivity <= 0:
        raise ValueError("diffusivity must be positive")
    concentration=equilibrium(totals[None,:])[0]
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
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or np.any(t < 0) or np.any(np.diff(t) < 0):
        raise ValueError("t must be increasing and nonnegative")
    if int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
        raise ValueError("harmonic must be an integer from 1 through 8")
    if t.size == 0:
        return np.empty((0, 2, 2), dtype=float)
    covariance,mobility=fluctuation_matrices(totals,diffusivity)
    relaxation=(int(harmonic)*np.pi/.001)**2*mobility@np.linalg.inv(covariance)
    initial_covariance=np.diag(totals)
    return np.array([expm(-float(time)*relaxation)@initial_covariance for time in t])
