import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
from scipy.sparse import diags

# Neutral four-ion association A+ + B+ + 2 C- <-> P.
ASSOCIATION=0.5
NEUTRAL_MOBILITY=0.15

def equilibrium(total):
    total = np.asarray(total, dtype=float)
    if total.ndim != 2 or total.shape[1] != 2:
        raise ValueError("total must have shape (n, 2)")
    a, b = total.T
    bound = np.zeros_like(a)
    for _ in range(24):
        free_a = a - bound
        free_b = b - bound
        anion = free_a + free_b
        value = bound - ASSOCIATION * free_a * free_b * anion ** 2
        derivative = 1 + ASSOCIATION * (anion ** 3 + 4 * free_a * free_b * anion)
        change = value / derivative
        bound = np.clip(bound - change, 0.0, np.minimum(a, b))
        if np.max(np.abs(change)) < 1e-13:
            break
    return np.column_stack((a - bound, b - bound, a + b - 2 * bound, bound))

class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self,data):
        """Fit the common diffusion scale from the single-cation records.

        A preparation containing only A (or only B) remains unassociated and
        is a scalar diffusion problem.  Eliminating the zero-current electric
        field gives effective diffusivities 4D/3 and 8D/3, respectively.  The
        cosine modes of the no-flux Laplacian therefore make evaluating the
        weighted least-squares objective both quick and numerically stable.
        """
        x = np.asarray(data["x"], dtype=float)
        times = np.asarray(data["t"], dtype=float)
        initial = np.asarray(data["initial"], dtype=float)
        observed = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)

        if x.ndim != 1 or len(x) < 2 or not np.all(np.diff(x) > 0):
            raise ValueError("x must be an increasing one-dimensional mesh")
        if times.ndim != 1 or np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("t must be an increasing nonnegative array")
        if initial.ndim != 3 or initial.shape[-1] != 2:
            raise ValueError("initial must have shape (preparation, x, 2)")

        dx = np.diff(x)
        if not np.allclose(dx, dx[0], rtol=1e-8, atol=0):
            raise ValueError("x must be equally spaced")
        n = len(x)
        eigenvalues = 4.0 / dx[0] ** 2 * np.sin(
            np.arange(n) * np.pi / (2.0 * (n - 1))
        ) ** 2

        # Calibration preparations are documented to contain one cation
        # constituent only.  Handle either channel without relying on exact
        # floating-point zeros in the input.
        channel_data = []
        for preparation in range(initial.shape[0]):
            scale = np.ptp(initial[preparation], axis=0)
            channel = int(np.argmax(scale))
            if scale[channel] == 0 or scale[1 - channel] > 1e-10 * max(1.0, scale[channel]):
                continue
            coefficients = dct(initial[preparation, :, channel], type=1)
            mode_factor = (4.0 / 3.0, 8.0 / 3.0)[channel]
            channel_data.append((coefficients, mode_factor,
                                 observed[preparation, :, :, channel],
                                 sigma[preparation, :, :, channel]))
        if not channel_data:
            raise ValueError("calibration data contain no single-cation preparation")

        def objective(log_diffusivity):
            diffusivity = np.exp(log_diffusivity)
            score = 0.0
            for coefficients, mode_factor, values, errors in channel_data:
                prediction = idct(
                    np.exp(-times[:, None] * diffusivity * mode_factor * eigenvalues[None, :])
                    * coefficients[None, :], type=1, axis=1
                )
                score += np.sum(((prediction - values) / errors) ** 2)
            return score

        result = minimize_scalar(
            objective, bounds=(np.log(1e-14), np.log(1e-6)), method="bounded",
            options={"xatol": 1e-12},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError("unable to fit diffusivity")
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self,t,x,initial):
        t,x,initial=np.asarray(t, dtype=float),np.asarray(x, dtype=float),np.asarray(initial, dtype=float)
        count=len(x)-1
        spatial_rate=4/(x[1]-x[0])**2*np.sin(np.arange(count+1)*np.pi/(2*count))**2
        coefficients=dct(initial,type=1,axis=0)
        rates=spatial_rate[:,None]*self.diffusivity*np.array([4/3,8/3])[None,:]
        return idct(np.exp(-t[:,None,None]*rates[None,:,:])*coefficients[None,:,:],type=1,axis=1)

    def predict(self,t,x,initial):
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if t.ndim != 1 or np.any(t < 0) or np.any(np.diff(t) < 0):
            raise ValueError("t must be an increasing nonnegative array")
        if x.ndim != 1 or len(x) < 2 or not np.all(np.diff(x) > 0):
            raise ValueError("x must be an increasing one-dimensional mesh")
        if initial.shape != (len(x), 2):
            raise ValueError("initial must have shape (len(x), 2)")
        if not np.allclose(np.diff(x), x[1] - x[0], rtol=1e-8, atol=0):
            raise ValueError("x must be equally spaced")
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise ValueError("fit the model before prediction")
        if len(t) == 0:
            return np.empty((0, len(x), 2), dtype=float)
        if t[-1] == 0:
            return initial[None,:,:].copy()
        if np.all(initial[:,0]==0) or np.all(initial[:,1]==0):
            return self._binary(t,x,initial)
        count=len(x);dx=x[1]-x[0]
        width=np.full(count,dx);width[[0,-1]]*=.5
        da,db,dc=self.diffusivity*np.array([1.,4.,2.])
        def derivative(time,flat):
            c=equilibrium(flat.reshape(count,2))
            face=(c[:-1]+c[1:])/2
            gradient=np.diff(c,axis=0)/dx
            denominator = da * face[:, 0] + db * face[:, 1] + dc * face[:, 2]
            numerator = da * gradient[:, 0] + db * gradient[:, 1] - dc * gradient[:, 2]
            potential_gradient = np.divide(
                -numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
            )
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
    totals = np.asarray(totals, dtype=float)
    if totals.shape != (2,) or not np.all(np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("totals must contain two positive finite concentrations")
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
    if t.ndim != 1 or np.any(~np.isfinite(t)) or np.any(t < 0) or np.any(np.diff(t) < 0):
        raise ValueError("t must be an increasing nonnegative array")
    if int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
        raise ValueError("harmonic must be an integer from 1 through 8")
    covariance,mobility=fluctuation_matrices(totals,diffusivity)
    relaxation=(int(harmonic)*np.pi/.001)**2*mobility@np.linalg.inv(covariance)
    initial_covariance=np.diag(np.asarray(totals, dtype=float))
    result = np.empty((len(t), 2, 2), dtype=float)
    for index, time in enumerate(t):
        result[index] = expm(-float(time)*relaxation)@initial_covariance
    return result

def correlate(self,t,totals,harmonic=1):
    if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
        raise ValueError("fit the model before correlation prediction")
    return equilibrium_correlations(t, totals, harmonic, self.diffusivity)

TransportModel.correlate=correlate
