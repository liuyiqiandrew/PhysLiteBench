from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

WIDTH, HEIGHT = .012, .008
DENSITY, CONDUCTIVITY = 1000., 1e5
# Comparable spatial resolution along the two physical dimensions.
X_MODES, Y_MODES = 64, 48


def derivative_overlap(target, source):
    """Integral of normalized sine(target) times derivative of sine(source)."""
    p, m = np.asarray(target, float)[:, None], np.asarray(source, float)[None, :]
    denominator = p*p-m*m
    return np.divide(4*p*m, denominator, out=np.zeros_like(denominator),
                     where=((p+m) % 2 == 1))


@lru_cache(4)
def operators(nx=X_MODES, ny=Y_MODES):
    """Galerkin operators in the area-normalized basis 2 sin(m pi x/W)
    sin(n pi y/H). Only modes invariant under a half turn are forced.

    Write the in-plane current as j=(psi_y, -psi_x). Insulating walls
    make psi constant on the connected boundary (chosen to be zero).
    Ohm's law gives -Laplacian(psi) = sigma * (B dot grad)u, so the
    magnetic dissipation is sigma * D.T @ (-Laplacian)^-1 @ D.
    """
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

    # For <dx u_i, (-Laplacian)^-1 dy u_j>, the electric mode is
    # (m_j, n_i). Unlike the squared-field terms, this couples odd/odd
    # velocity modes to even/even modes and retains the field's sign.
    cross = (derivative_overlap(m, m).T/WIDTH
             * derivative_overlap(n, n)/HEIGHT
             / ((m[None, :]*np.pi/WIDTH)**2
                + (n[:, None]*np.pi/HEIGHT)**2))
    qxy = cross + cross.T

    return m, n, wave2, mean, qx, qy, qxy


@lru_cache(16)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy, qxy = operators(nx, ny)
    magnetic = bx*bx*qx+by*by*qy+bx*by*qxy
    operator = np.diag(viscosity/DENSITY*wave2)+CONDUCTIVITY/DENSITY*magnetic
    if bx == 0.:
        groups = [np.flatnonzero((m == fixed) & (n % 2 == 1)) for fixed in np.unique(m[m % 2 == 1])]
    elif by == 0.:
        groups = [np.flatnonzero((n == fixed) & (m % 2 == 1)) for fixed in np.unique(n[n % 2 == 1])]
    else:
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
        """Estimate the common viscosity by Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required')
        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], float)
        errors = np.asarray([record['sigma'] for record in records], float)
        if not (np.isfinite(values).all() and np.isfinite(errors).all()
                and (errors > 0).all()):
            raise ValueError('Calibration values must be finite and sigma positive')

        def objective(viscosity):
            residual = (self._predict(experiments, viscosity)-values)/errors
            return float(residual @ residual)

        result = minimize_scalar(objective, bounds=(0.006, 0.018),
                                 method='bounded', options={'xatol': 1e-10})
        if not result.success:
            raise RuntimeError('Viscosity calibration failed: ' + result.message)
        # Include endpoints, since the constrained optimum can be on a bound.
        candidates = [(float(result.fun), float(result.x))]
        candidates.extend((objective(v), v) for v in (0.006, 0.018))
        self.viscosity = min(candidates)[1]
        return self

    def predict(self, experiments):
        """Return mean or point velocities in m/s after a pressure step."""
        if self.viscosity is None:
            raise ValueError('Call fit before predict')
        return self._predict(experiments, self.viscosity)

    @staticmethod
    def _predict(experiments, viscosity):
        m, n, _, mean, *_ = operators()
        output = []
        for e in experiments:
            if e['time'] == 0 or e['pressure_gradient'] == 0:
                output.append(0.)
                continue
            bx, by = e['magnetic_x'], e['magnetic_y']
            observation = mean if e['observable'] == 'mean' else 2*np.sin(m*np.pi*e['x'])*np.sin(n*np.pi*e['y'])
            value = 0.
            for ids, rates, vectors, forcing in spectrum(viscosity, bx, by):
                amplitude = vectors@(-np.expm1(-rates*e['time'])/rates*forcing)
                value += observation[ids]@amplitude
            output.append(e['pressure_gradient']/DENSITY*value)
        return np.asarray(output, dtype=float)
