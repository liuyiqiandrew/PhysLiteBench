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

    # The current streamfunction vanishes at the insulating boundary.  Its
    # source is B . grad(u), so an oblique field also has a mixed term.
    dx = derivative_overlap(m, m)/WIDTH
    dy = derivative_overlap(n, n)/HEIGHT
    electric_wave2 = (np.pi*m[None, :]/WIDTH)**2 + (np.pi*n[:, None]/HEIGHT)**2
    mixed = dx.T*dy/electric_wave2
    return m, n, wave2, mean, qx, qy, mixed + mixed.T


@lru_cache(64)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy, mixed = operators(nx, ny)
    magnetic = bx*bx*qx+by*by*qy+bx*by*mixed
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
        records = list(records)
        experiments = [r['input'] for r in records]
        measured = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)

        def objective(viscosity):
            self.viscosity = float(viscosity)
            residual = (self.predict(experiments)-measured)/sigma
            return float(residual@residual)

        result = minimize_scalar(objective, bounds=(.006, .018),
                                 method='bounded',
                                 options={'xatol': 1e-9})
        self.viscosity = float(result.x)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise ValueError('Fit the model before prediction')
        experiments = list(experiments)
        output = np.empty(len(experiments), dtype=float)
        grouped = {}
        for i, e in enumerate(experiments):
            grouped.setdefault((float(e['magnetic_x']),
                                float(e['magnetic_y'])), []).append(i)

        m, n, _, mean, *_ = operators()
        for (bx, by), indices in grouped.items():
            times = np.asarray([experiments[i]['time'] for i in indices])
            observations = np.stack([
                mean if experiments[i]['observable'] == 'mean' else
                2*np.sin(np.pi*m*experiments[i]['x'])*
                np.sin(np.pi*n*experiments[i]['y'])
                for i in indices])
            velocities = np.zeros(len(indices))
            for ids, rates, vectors, forcing in spectrum(self.viscosity, bx, by):
                response = -np.expm1(-times[:, None]*rates[None, :])/rates
                projected = observations[:, ids]@vectors
                velocities += np.sum(projected*forcing[None, :]*response, axis=1)
            gradients = np.asarray([experiments[i]['pressure_gradient']
                                    for i in indices])
            output[indices] = gradients/DENSITY*velocities
        return output
