"""Transient, quasi-static MHD flow with insulating, no-slip duct walls.

Write the transverse current as j = (psi_y, -psi_x). Insulation makes
psi constant on the connected boundary (chosen to be zero), and Ohm's law
gives -laplacian(psi) = sigma * (B dot grad) u. Eliminating psi gives the
positive magnetic damping operator D.T * (-laplacian)^-1 * D, where
D = B dot grad. A sine Galerkin basis enforces both wall conditions.
"""

from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

WIDTH, HEIGHT = .012, .008
DENSITY, CONDUCTIVITY = 1000., 1e5
# More modes along the wider dimension resolve the thin magnetic wall layers.
X_MODES, Y_MODES = 64, 48


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

    # Mixed derivatives share exactly one electric mode: (m_j, n_i).
    # They couple odd/odd velocity modes to even/even modes. Both are
    # invariant under a half turn about the duct centre; no other parity
    # is forced by the spatially uniform pressure gradient.
    dx = derivative_overlap(m, m)/WIDTH
    dy = derivative_overlap(n, n)/HEIGHT
    mixed = dx.T*dy/((m[None, :]*np.pi/WIDTH)**2
                      +(n[:, None]*np.pi/HEIGHT)**2)
    qxy = mixed+mixed.T
    return m, n, wave2, mean, qx, qy, qxy


@lru_cache(8)
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
        experiments = [r['input'] for r in records]
        measured = np.asarray([r['value'] for r in records], dtype=float)
        errors = np.asarray([r['sigma'] for r in records], dtype=float)
        if not np.all(np.isfinite(errors) & (errors > 0)):
            raise ValueError('Measurement standard deviations must be positive and finite')

        def objective(viscosity):
            self.viscosity = float(viscosity)
            residual = (self.predict(experiments)-measured)/errors
            return float(residual@residual)

        optimum = minimize_scalar(objective, bounds=(0.006, 0.018),
                                  method='bounded', options={'xatol': 1e-10})
        # Include endpoints, since the constrained optimum may lie there.
        self.viscosity = min((float(optimum.x), 0.006, 0.018), key=objective)
        return self

    def predict(self, experiments):
        """Return point or area-mean axial velocities in m/s."""
        if self.viscosity is None:
            raise ValueError('Fit the model before predicting')
        m, n, _, mean, *_ = operators()
        output = []
        for e in experiments:
            if e['time'] == 0 or e['pressure_gradient'] == 0:
                output.append(0.)
                continue
            bx, by = e['magnetic_x'], e['magnetic_y']
            # Reversing the entire field leaves the flow unchanged. Reversing
            # just one component reflects the profile across one mid-plane.
            # Use those symmetries to share spectra between signed fields.
            if e['observable'] == 'mean':
                observation = mean
            else:
                x = 1.-e['x'] if bx*by < 0 else e['x']
                observation = 2*np.sin(m*np.pi*x)*np.sin(n*np.pi*e['y'])
            value = 0.
            for ids, rates, vectors, forcing in spectrum(self.viscosity, abs(bx), abs(by)):
                amplitude = vectors@(-np.expm1(-rates*e['time'])/rates*forcing)
                value += observation[ids]@amplitude
            output.append(e['pressure_gradient']/DENSITY*value)
        return np.array(output)
