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
        """Estimate the (common) dynamic viscosity from noisy observations.

        The records have independent Gaussian errors, so their maximum
        likelihood estimate is the minimum of the weighted squared residuals.
        There is only one unknown parameter and its physical range is known,
        which makes a bounded scalar minimisation both more reliable and less
        prone to scaling issues than a general purpose curve fitter.
        """
        records = list(records)
        if not records:
            # There is no statistical information in an empty calibration
            # set.  Keep the model usable while respecting the specified
            # physical interval.
            self.viscosity = 0.012
            return self

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        uncertainty = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(observed).all() or not np.isfinite(uncertainty).all()
                or np.any(uncertainty <= 0)):
            raise ValueError('calibration values must be finite and sigma must be positive')

        # predict() uses self.viscosity, so keep the trial value local to this
        # closure and restore a valid value even if the optimizer stops early.
        previous = self.viscosity

        def objective(viscosity):
            self.viscosity = float(viscosity)
            residual = (self.predict(experiments) - observed) / uncertainty
            return float(residual @ residual)

        try:
            result = minimize_scalar(
                objective, bounds=(0.006, 0.018), method='bounded',
                options={'xatol': 1.e-10},
            )
            # ``bounded`` need not inspect an endpoint.  Including them also
            # handles data whose likelihood is monotone across the allowed
            # physical range.
            candidates = (0.006, float(result.x), 0.018)
            self.viscosity = min(candidates, key=objective)
        except Exception:
            self.viscosity = previous
            raise
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('fit must be called before predict')
        if not np.isfinite(self.viscosity) or not 0.006 <= self.viscosity <= 0.018:
            raise ValueError('viscosity must be finite and lie in [0.006, 0.018]')

        m, n, _, mean, *_ = operators()
        output = []
        for e in experiments:
            bx, by = float(e['magnetic_x']), float(e['magnetic_y'])
            time = float(e['time'])
            if e['observable'] == 'mean':
                observation = mean
            elif e['observable'] == 'point':
                observation = (2*np.sin(m*np.pi*float(e['x']))
                               * np.sin(n*np.pi*float(e['y'])))
            else:
                raise ValueError("observable must be 'mean' or 'point'")
            value = 0.
            for ids, rates, vectors, forcing in spectrum(self.viscosity, bx, by):
                # expm1 maintains relative accuracy in the short-time limit.
                amplitude = vectors@(-np.expm1(-rates*time)/rates*forcing)
                value += observation[ids]@amplitude
            output.append(float(e['pressure_gradient'])/DENSITY*value)
        return np.asarray(output, dtype=float)
