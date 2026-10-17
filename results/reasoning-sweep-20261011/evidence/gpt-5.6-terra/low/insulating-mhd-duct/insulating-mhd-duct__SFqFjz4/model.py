from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

WIDTH, HEIGHT = .012, .008
DENSITY, CONDUCTIVITY = 1000., 1e5
X_MODES, Y_MODES = 40, 56


def derivative_overlap(target, source):
    """Integral of normalized sine(target) times derivative of sine(source)."""
    p, m = np.asarray(target, float)[:, None], np.asarray(source, float)[None, :]
    denominator = p*p-m*m
    return np.divide(4*p*m, denominator, out=np.zeros_like(denominator),
                     where=((p+m) % 2 == 1))


@lru_cache(4)
def operators(nx=X_MODES, ny=Y_MODES):
    m, n = np.meshgrid(np.arange(1, nx+1), np.arange(1, ny+1), indexing='ij')
    keep = (m+n) % 2 == 0
    m, n = m[keep], n[keep]
    wave2 = (m*np.pi/WIDTH)**2+(n*np.pi/HEIGHT)**2
    mean = np.where(m % 2 == 1, 8/(np.pi*np.pi*m*n), 0.)
    qx, qy = np.zeros((len(m), len(m))), np.zeros((len(m), len(m)))
    electric = np.arange(1, 401)
    for fixed in np.unique(n):
        ids = np.flatnonzero(n == fixed)
        d = derivative_overlap(electric, m[ids])/WIDTH
        k2 = (electric*np.pi/WIDTH)**2+(fixed*np.pi/HEIGHT)**2
        qx[np.ix_(ids, ids)] = d.T@(d/k2[:, None])
    for fixed in np.unique(m):
        ids = np.flatnonzero(m == fixed)
        d = derivative_overlap(electric, n[ids])/HEIGHT
        k2 = (fixed*np.pi/WIDTH)**2+(electric*np.pi/HEIGHT)**2
        qy[np.ix_(ids, ids)] = d.T@(d/k2[:, None])

    return m, n, wave2, mean, qx, qy


@lru_cache(64)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy = operators(nx, ny)
    magnetic = bx*bx*qx+by*by*qy
    operator = np.diag(viscosity/DENSITY*wave2)+CONDUCTIVITY/DENSITY*magnetic
    if bx == 0.:
        groups = [np.flatnonzero((m == fixed) & (n % 2 == 1)) for fixed in np.unique(m[m % 2 == 1])]
    elif by == 0.:
        groups = [np.flatnonzero((n == fixed) & (m % 2 == 1)) for fixed in np.unique(n[n % 2 == 1])]
    else:
        groups = [np.flatnonzero(m % 2 == 1)]
    result = []
    for ids in groups:
        values, vectors = eigh(operator[np.ix_(ids, ids)], check_finite=False)
        result.append((ids, values, vectors, vectors.T@mean[ids]))
    return result


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Estimate the (shared) viscosity by weighted least squares.

        The supplied standard deviations describe independent Gaussian noise,
        so minimizing this quantity is also the maximum-likelihood estimate.
        Materializing ``records`` also makes a generator input safe to use
        during the several objective evaluations made by the optimizer.
        """
        records = list(records)
        if not records:
            raise ValueError("fit requires at least one calibration record")

        experiments = [record['input'] for record in records]
        measured = np.asarray([record['value'] for record in records], dtype=float)
        uncertainty = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(measured).all() or not np.isfinite(uncertainty).all()
                or np.any(uncertainty <= 0)):
            raise ValueError("calibration values must be finite and sigmas positive")

        def objective(viscosity):
            # Assign temporarily because predict is the single implementation
            # of the forward model used both here and after fitting.
            self.viscosity = float(viscosity)
            residual = (self.predict(experiments) - measured) / uncertainty
            return float(residual @ residual)

        result = minimize_scalar(
            objective, bounds=(0.006, 0.018), method='bounded',
            options={'xatol': 1e-11},
        )
        if not result.success or not np.isfinite(result.fun):
            self.viscosity = None
            raise RuntimeError("unable to fit viscosity")
        self.viscosity = float(result.x)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("call fit before predict")
        m, n, _, mean, *_ = operators()
        output = []
        for e in experiments:
            bx, by = e['magnetic_x'], e['magnetic_y']
            observation = mean if e['observable'] == 'mean' else 2*np.sin(m*np.pi*e['x'])*np.sin(n*np.pi*e['y'])
            value = 0.
            for ids, rates, vectors, forcing in spectrum(self.viscosity, bx, by):
                amplitude = vectors@(-np.expm1(-rates*e['time'])/rates*forcing)
                value += observation[ids]@amplitude
            output.append(e['pressure_gradient']/DENSITY*value)
        return np.array(output)
