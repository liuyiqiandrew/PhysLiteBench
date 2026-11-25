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

    # The two components of B do not merely add when both are nonzero.
    # Eliminating the electric potential gives
    #
    #   (B_x D_x + B_y D_y)^T (-laplacian)^-1 (B_x D_x + B_y D_y).
    #
    # qx and qy above are the two square terms.  The mixed term couples the
    # odd-odd velocity modes (which pressure directly drives) to even-even
    # modes.  It is therefore invisible in one-component-field calibration
    # runs, but is important for a general applied field.
    qxy = np.zeros_like(qx)
    for i, (mi, ni) in enumerate(zip(m, n)):
        ids = np.flatnonzero(((m + mi) % 2 == 1) & ((n + ni) % 2 == 1))
        if not len(ids):
            continue
        mj, nj = m[ids], n[ids]
        dx = derivative_overlap(mj, np.array([mi])).ravel() / WIDTH
        dy = derivative_overlap(np.array([ni]), nj).ravel() / HEIGHT
        k2 = (mj*np.pi/WIDTH)**2 + (ni*np.pi/HEIGHT)**2
        qxy[i, ids] = dx * dy / k2

    return m, n, wave2, mean, qx, qy, qxy


@lru_cache(64)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy, qxy = operators(nx, ny)
    magnetic = bx*bx*qx + by*by*qy + bx*by*(qxy + qxy.T)
    operator = np.diag(viscosity/DENSITY*wave2)+CONDUCTIVITY/DENSITY*magnetic
    if bx == 0.:
        groups = [np.flatnonzero((m == fixed) & (n % 2 == 1)) for fixed in np.unique(m[m % 2 == 1])]
    elif by == 0.:
        groups = [np.flatnonzero((n == fixed) & (m % 2 == 1)) for fixed in np.unique(n[n % 2 == 1])]
    else:
        # With both components present, the mixed term connects odd-odd and
        # even-even modes, so this whole parity sector must be retained.
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
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        errors = np.asarray([record['sigma'] for record in records], dtype=float)
        if not (np.isfinite(values).all() and np.isfinite(errors).all()
                and np.all(errors > 0)):
            raise ValueError("calibration values must be finite and sigmas positive")

        # The readings have independently supplied Gaussian uncertainties, so
        # maximum likelihood is weighted least squares.  Keeping the trial
        # viscosity on self lets predict use the same numerical path as users.
        def objective(viscosity):
            self.viscosity = float(viscosity)
            residual = (self.predict(experiments) - values) / errors
            return float(residual @ residual)

        result = minimize_scalar(
            objective, bounds=(0.006, 0.018), method='bounded',
            options={'xatol': 1e-11},
        )
        self.viscosity = float(result.x)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise ValueError("fit must be called before predict")
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
