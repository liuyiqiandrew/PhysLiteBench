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

    # For an oblique field the two transverse derivatives also have a
    # cross-term.  The electric-current projection is most simply written in
    # terms of the Dirichlet stream function: if D_x and D_y are derivative
    # overlaps from a velocity sine mode to a stream-function sine mode, the
    # magnetic damping matrix is D.T @ laplace_inverse @ D.  The diagonal
    # pieces are qx and qy above.  A common stream-function mode in the two
    # mixed products fixes one of its indices to a velocity index, giving the
    # two terms below without constructing a large four-index array.
    dx = derivative_overlap(m, m)/WIDTH
    dy = derivative_overlap(n, n)/HEIGHT
    x_source_y_target = (m[None, :]*np.pi/WIDTH)**2 + (n[:, None]*np.pi/HEIGHT)**2
    x_target_y_source = (m[:, None]*np.pi/WIDTH)**2 + (n[None, :]*np.pi/HEIGHT)**2
    cross = (dx.T*dy/x_source_y_target + dx*dy.T/x_target_y_source)

    return m, n, wave2, mean, qx, qy, cross


@lru_cache(64)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy, cross = operators(nx, ny)
    magnetic = bx*bx*qx+by*by*qy+bx*by*cross
    operator = np.diag(viscosity/DENSITY*wave2)+CONDUCTIVITY/DENSITY*magnetic
    if bx == 0.:
        groups = [np.flatnonzero((m == fixed) & (n % 2 == 1)) for fixed in np.unique(m[m % 2 == 1])]
    elif by == 0.:
        groups = [np.flatnonzero((n == fixed) & (m % 2 == 1)) for fixed in np.unique(n[n % 2 == 1])]
    else:
        groups = [np.flatnonzero((m+n) % 2 == 0)]
    result = []
    for ids in groups:
        values, vectors = eigh(operator[np.ix_(ids, ids)], check_finite=False)
        result.append((ids, values, vectors, vectors.T@mean[ids]))
    return result


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit the dynamic viscosity to calibration observations.

        The measurement errors are independent Gaussians with known standard
        deviations, so fitting is a one-dimensional weighted least-squares
        problem.  Keeping the fit in viscosity (rather than, for example,
        fitting an unconstrained logarithm) also makes the documented physical
        bounds explicit.
        """
        inputs = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        if not len(inputs):
            raise ValueError('at least one calibration record is required')
        if not (np.isfinite(values).all() and np.isfinite(sigmas).all()):
            raise ValueError('calibration values and sigmas must be finite')
        if np.any(sigmas <= 0):
            raise ValueError('calibration sigmas must be positive')

        def objective(viscosity):
            self.viscosity = float(viscosity)
            prediction = self.predict(inputs)
            residual = (prediction - values) / sigmas
            return float(residual @ residual)

        result = minimize_scalar(
            objective,
            bounds=(0.006, 0.018),
            method='bounded',
            options={'xatol': 1e-12},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError('viscosity fit did not converge')

        self.viscosity = float(np.clip(result.x, 0.006, 0.018))
        return self

    def predict(self, experiments):
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
