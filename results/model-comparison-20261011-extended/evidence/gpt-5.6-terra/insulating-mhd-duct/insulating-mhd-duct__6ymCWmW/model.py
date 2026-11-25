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


@lru_cache(4)
def cross_operator(nx=X_MODES, ny=Y_MODES):
    """Mixed x/y part of the insulating-wall magnetic resistance.

    For a transverse field, the divergence-free current is driven by
    ``B_x u_x + B_y u_y``.  The squared norm of that quantity after the
    Dirichlet inverse Laplacian produces this mixed term.  It changes both
    sine parities, so it is only relevant when both field components exist.
    """
    m, n, *_ = operators(nx, ny)
    dx = derivative_overlap(m, m) / WIDTH
    dy = derivative_overlap(n, n) / HEIGHT
    denominator = ((n[:, None] * np.pi / HEIGHT) ** 2
                   + (m[None, :] * np.pi / WIDTH) ** 2)
    return dx.T * dy / denominator


@lru_cache(64)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy = operators(nx, ny)
    magnetic = bx*bx*qx + by*by*qy
    if bx != 0. and by != 0.:
        mixed = cross_operator(nx, ny)
        magnetic = magnetic + bx*by*(mixed + mixed.T)
    operator = np.diag(viscosity/DENSITY*wave2)+CONDUCTIVITY/DENSITY*magnetic
    if bx == 0.:
        groups = [np.flatnonzero((m == fixed) & (n % 2 == 1)) for fixed in np.unique(m[m % 2 == 1])]
    elif by == 0.:
        groups = [np.flatnonzero((n == fixed) & (m % 2 == 1)) for fixed in np.unique(n[n % 2 == 1])]
    else:
        # The mixed resistance term takes odd/odd forcing into even/even
        # modes.  Both families are in the retained m+n-even basis.
        groups = [np.arange(len(m))]
    result = []
    for ids in groups:
        values, vectors = eigh(operator[np.ix_(ids, ids)], check_finite=False)
        result.append((ids, values, vectors, vectors.T@mean[ids]))
    return result


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit the common dynamic viscosity by weighted least squares.

        The calibration uncertainty is supplied with each measurement, so the
        appropriate objective is the Gaussian negative log likelihood (up to a
        constant).  All other quantities in an experiment are exact.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        measured = np.asarray([record['value'] for record in records], dtype=float)
        uncertainty = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(measured).all() or not np.isfinite(uncertainty).all()
                or np.any(uncertainty <= 0.0)):
            raise ValueError("calibration values and sigmas must be finite, with sigma > 0")

        def objective(viscosity):
            # Assigning here lets predict use exactly the same forward model as
            # callers do after fitting.
            self.viscosity = float(viscosity)
            residual = (self.predict(experiments) - measured) / uncertainty
            return float(residual @ residual)

        result = minimize_scalar(
            objective, bounds=(0.006, 0.018), method='bounded',
            options={'xatol': 1e-11},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("viscosity calibration did not converge")
        self.viscosity = float(result.x)
        return self

    def predict(self, experiments):
        if self.viscosity is None or not np.isfinite(self.viscosity):
            raise RuntimeError("fit must be called before predict")
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
