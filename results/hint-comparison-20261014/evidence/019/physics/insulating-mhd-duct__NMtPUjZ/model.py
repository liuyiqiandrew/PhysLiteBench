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


def _cosine_sine_overlap(cosine, sine):
    """Overlap of normalized cosine and sine modes on one interval.

    The cosine mode numbered zero is the constant Neumann mode.  This
    normalization is useful here because the electric potential has
    Neumann, rather than no-slip, boundary conditions.
    """
    cosine = np.asarray(cosine, dtype=int)
    result = np.zeros(cosine.shape, dtype=float)

    if sine % 2:
        result[cosine == 0] = 2. * np.sqrt(2.) / (np.pi * sine)

    nonzero = cosine != 0
    different_parity = (cosine + sine) % 2 == 1
    use = nonzero & different_parity
    # The parity condition makes the denominator nonzero wherever it is
    # used.  Keeping the assignment masked also avoids spurious warnings.
    result[use] = (4. * sine /
                   (np.pi * (sine*sine - cosine[use]*cosine[use])))
    return result


@lru_cache(4)
def oblique_coupling(nx=X_MODES, ny=Y_MODES):
    """The mixed part of the magnetic damping operator.

    For u = u(x,y) e_z, Ohm's law gives

        j = sigma (-grad(phi) + u cross B),

    while phi solves one Neumann Poisson problem.  The two terms in
    div(u cross B) therefore share that problem.  ``coupling`` is the
    coefficient matrix G_x K G_y; the full mixed term is its sum with its
    transpose.  It connects the odd-odd and even-even sine subspaces and
    is consequently absent from the axis-only responses.
    """
    m, n, _, _, _, _ = operators(nx, ny)
    size = len(m)
    coupling = np.zeros((size, size), dtype=float)

    for row, (mr, nr) in enumerate(zip(m, n)):
        # G_x reaches potentials with p=mr and q of parity opposite nr.
        # G_y reaches the same potentials from modes with m of parity
        # opposite mr and q=n_col.
        columns = np.flatnonzero(
            ((m % 2) != (mr % 2)) & ((n % 2) != (nr % 2)))
        if not len(columns):
            continue

        q = n[columns]
        gx = -(mr*np.pi/WIDTH) * _cosine_sine_overlap(q, int(nr))
        gy = -(q*np.pi/HEIGHT) * np.array([
            _cosine_sine_overlap(np.array([mr]), int(mc))[0]
            for mc in m[columns]
        ])
        denominator = ((mr*np.pi/WIDTH)**2 +
                       (q*np.pi/HEIGHT)**2)
        coupling[row, columns] = gx * gy / denominator

    return coupling


@lru_cache(64)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy = operators(nx, ny)
    if bx != 0. and by != 0.:
        # The mixed term is important for an oblique field.  Keeping the
        # even-even modes is necessary because it is the subspace coupled
        # to the pressure-driven odd-odd modes by that term.
        mixed = oblique_coupling(nx, ny)
        magnetic = bx*bx*qx + by*by*qy + bx*by*(mixed + mixed.T)
        ids = np.arange(len(m))
        groups = [ids]
    else:
        # With one field component, parity prevents odd-odd modes from
        # coupling to even-even modes.  Use the smaller independent blocks.
        ids = np.flatnonzero((m % 2 == 1) & (n % 2 == 1))
        if bx == 0.:
            fixed_values = np.unique(m[ids])
            groups = [ids[m[ids] == fixed] for fixed in fixed_values]
        else:
            fixed_values = np.unique(n[ids])
            groups = [ids[n[ids] == fixed] for fixed in fixed_values]
        magnetic = bx*bx*qx + by*by*qy

    operator = np.diag(viscosity/DENSITY*wave2)+CONDUCTIVITY/DENSITY*magnetic
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
            raise ValueError('records must not be empty')

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], float)
        deviations = np.asarray([record['sigma'] for record in records], float)
        if (not np.isfinite(values).all() or
                not np.isfinite(deviations).all() or
                (deviations <= 0.).any()):
            raise ValueError('calibration values and sigma must be finite')

        def objective(viscosity):
            prediction = self._predict(experiments, viscosity)
            residual = (prediction - values) / deviations
            return float(residual @ residual)

        result = minimize_scalar(objective, bounds=(.006, .018),
                                 method='bounded',
                                 options={'xatol': 1.e-11, 'maxiter': 100})
        # The bounded optimizer should always succeed for this smooth
        # one-parameter problem.  Clipping also keeps the public attribute
        # within the documented interval in case scipy stops at an endpoint.
        self.viscosity = float(np.clip(result.x, .006, .018))
        return self

    def predict(self, experiments):
        if self.viscosity is None or not np.isfinite(self.viscosity):
            raise ValueError('fit must be called before predict')
        return self._predict(experiments, self.viscosity)

    def _predict(self, experiments, viscosity):
        m, n, _, mean, *_ = operators()
        output = []
        for e in experiments:
            bx, by = e['magnetic_x'], e['magnetic_y']
            observation = mean if e['observable'] == 'mean' else 2*np.sin(m*np.pi*e['x'])*np.sin(n*np.pi*e['y'])
            value = 0.
            for ids, rates, vectors, forcing in spectrum(viscosity, bx, by):
                amplitude = vectors@(-np.expm1(-rates*e['time'])/rates*forcing)
                value += observation[ids]@amplitude
            output.append(e['pressure_gradient']/DENSITY*value)
        return np.array(output)
