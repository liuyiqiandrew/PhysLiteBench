"""Modal model for fully developed conducting flow in a rectangular duct.

The velocity has only an axial component, while the induced current is in the
cross-section.  The latter is obtained by projecting ``u x B`` onto fields
which are divergence free and have zero normal component at the walls.  This
is the Neumann Poisson problem for the electrostatic potential, and is the
part of the model which is important when both components of B are present.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


WIDTH, HEIGHT = .012, .008
DENSITY, CONDUCTIVITY = 1000., 1e5
X_MODES, Y_MODES = 40, 56
ELECTRIC_MODES = 400


def derivative_overlap(target, source):
    """Integral of normalized sine(target) times derivative of sine(source)."""
    p, m = np.asarray(target, float)[:, None], np.asarray(source, float)[None, :]
    denominator = p*p-m*m
    return np.divide(4*p*m, denominator, out=np.zeros_like(denominator),
                     where=((p+m) % 2 == 1))


def _sine_to_cosine_overlap(cosine_index, sine_index):
    """<normalized cosine, normalized sine> on a unit interval.

    The zero cosine mode is included.  It is easy to miss this mode in the
    Neumann potential, but it contributes to the current and hence to the
    magnetic damping.
    """
    q = np.asarray(cosine_index, dtype=float)[:, None]
    n = np.asarray(sine_index, dtype=float)[None, :]
    result = np.zeros((len(cosine_index), len(sine_index)), dtype=float)
    zero = q == 0
    result[zero[:, 0], :] = 2. * np.sqrt(2.) / (np.pi * n[0])
    nonzero = (~zero) & (((q + n) % 2) == 1)
    np.divide(4. * n, np.pi * (n*n-q*q), out=result, where=nonzero)
    result[zero[:, 0], :] = 2. * np.sqrt(2.) / (np.pi * n[0])
    return result


@lru_cache(maxsize=4)
def operators(nx=X_MODES, ny=Y_MODES):
    """Return the retained velocity modes and their modal operators.

    The pressure forcing is odd/odd.  An oblique field also couples those
    modes to the even/even sine modes through the cross term in the one
    electrostatic potential, so both parity sectors are retained.
    """
    mx = np.arange(1, nx + 1)
    ny_modes = np.arange(1, ny + 1)
    m, n = np.meshgrid(mx, ny_modes, indexing='ij')
    keep = (m + n) % 2 == 0
    m, n = m[keep], n[keep]

    wave2 = (m * np.pi / WIDTH)**2 + (n * np.pi / HEIGHT)**2
    mean = np.where((m % 2 == 1) & (n % 2 == 1),
                    8. / (np.pi*np.pi*m*n), 0.)

    damping_x = np.zeros((len(m), len(m)))
    damping_y = np.zeros_like(damping_x)

    # For d/dx, the x cosine index is exactly m; the y sine is expanded in
    # Neumann cosine modes of the opposite parity.  This gives independent
    # blocks at fixed m.
    for fixed_m in np.unique(m):
        ids = np.flatnonzero(m == fixed_m)
        q = (np.arange(0, ELECTRIC_MODES + 1, 2) if fixed_m % 2 else
             np.arange(1, ELECTRIC_MODES + 1, 2))
        overlap = _sine_to_cosine_overlap(q, n[ids])
        derivative = (fixed_m * np.pi / WIDTH) * overlap
        wave = (fixed_m * np.pi / WIDTH)**2 + (q * np.pi / HEIGHT)**2
        block = derivative.T @ (derivative / wave[:, None])
        damping_x[np.ix_(ids, ids)] = np.eye(len(ids)) - block

    # For d/dy, the y cosine index is exactly n; the x sine is expanded in
    # Neumann cosine modes of the opposite parity.  This gives independent
    # blocks at fixed n.
    for fixed_n in np.unique(n):
        ids = np.flatnonzero(n == fixed_n)
        p = (np.arange(0, ELECTRIC_MODES + 1, 2) if fixed_n % 2 else
             np.arange(1, ELECTRIC_MODES + 1, 2))
        overlap = _sine_to_cosine_overlap(p, m[ids])
        derivative = (fixed_n * np.pi / HEIGHT) * overlap
        wave = (p * np.pi / WIDTH)**2 + (fixed_n * np.pi / HEIGHT)**2
        block = derivative.T @ (derivative / wave[:, None])
        damping_y[np.ix_(ids, ids)] = np.eye(len(ids)) - block

    # The two components of u x B are combined before the force is formed.
    # For odd/odd versus even/even modes their potential modes overlap, giving
    # a nonzero cross Gram matrix.  It is added in spectrum() with the proper
    # Bx*By sign rather than treating the two axis responses independently.
    cross = np.zeros_like(damping_x)
    odd = (m % 2 == 1)
    odd_ids = np.flatnonzero(odd)
    even_ids = np.flatnonzero(~odd)
    if len(odd_ids) and len(even_ids):
        oi_m, oi_n = m[odd_ids], n[odd_ids]
        ei_m, ei_n = m[even_ids], n[even_ids]
        # The expression below is the direct shared-potential coefficient;
        # it is written as a matrix to retain all mode couplings.
        shared_p = oi_m[:, None]
        shared_q = ei_n[None, :]
        first = ((oi_m * np.pi / WIDTH)[:, None] *
                 _sine_to_cosine_overlap(shared_q.ravel(), oi_n).T)
        second = ((ei_n * np.pi / HEIGHT)[None, :] *
                  _sine_to_cosine_overlap(shared_p.ravel(), ei_m))
        wave = (shared_p * np.pi / WIDTH)**2 + (shared_q * np.pi / HEIGHT)**2
        block = first * second / wave

        # The other component pairing uses the potential mode
        # (p,q)=(even m, odd n): target odd uses d/dy and source even uses
        # d/dx.  Both pairings belong to the same total current.
        shared_p = ei_m[None, :]
        shared_q = oi_n[:, None]
        target_y = ((oi_n * np.pi / HEIGHT)[:, None] *
                    _sine_to_cosine_overlap(shared_p.ravel(), oi_m).T)
        source_x = ((ei_m * np.pi / WIDTH)[None, :] *
                    _sine_to_cosine_overlap(shared_q.ravel(), ei_n))
        wave = (shared_p * np.pi / WIDTH)**2 + (shared_q * np.pi / HEIGHT)**2
        block += target_y * source_x / wave
        cross[np.ix_(odd_ids, even_ids)] = block
        cross[np.ix_(even_ids, odd_ids)] = block.T

    return m, n, wave2, mean, damping_x, damping_y, cross


@lru_cache(maxsize=128)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    """Eigen-decomposition of the decay operator for one field."""
    m, n, wave2, mean, damping_x, damping_y, cross = operators(nx, ny)
    active = np.arange(len(m))
    if bx == 0. or by == 0.:
        active = np.flatnonzero((m % 2 == 1) & (n % 2 == 1))
    magnetic = (by*by*damping_x + bx*bx*damping_y + bx*by*cross)
    operator = ((viscosity / DENSITY) * np.diag(wave2) +
                (CONDUCTIVITY / DENSITY) * magnetic)
    operator = operator[np.ix_(active, active)]
    rates, vectors = eigh(operator, check_finite=False)
    return active, rates, vectors, vectors.T @ mean[active]


def _time_factor(rates, time):
    """Integral of exp(-rates*t) from zero to ``time`` without cancellation."""
    rates = np.asarray(rates)
    if time == 0:
        return np.zeros_like(rates)
    return -np.expm1(-rates * time) / rates


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        measured = np.asarray([r['value'] for r in records], dtype=float)
        uncertainty = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.isfinite(measured).all() or
                not np.isfinite(uncertainty).all() or
                np.any(uncertainty <= 0)):
            raise ValueError('calibration values and sigma must be finite')
        experiments = [r['input'] for r in records]

        def objective(viscosity):
            prediction = self._predict_with_viscosity(experiments, viscosity)
            residual = (prediction - measured) / uncertainty
            return float(residual @ residual)

        result = minimize_scalar(objective, bounds=(0.006, 0.018),
                                 method='bounded',
                                 options={'xatol': 2e-11, 'maxiter': 80})
        self.viscosity = float(np.clip(result.x, 0.006, 0.018))
        return self

    def _predict_with_viscosity(self, experiments, viscosity):
        m, n, _, mean, _, _, _ = operators()
        result = np.empty(len(experiments), dtype=float)
        cached = {}
        for index, experiment in enumerate(experiments):
            bx = float(experiment['magnetic_x'])
            by = float(experiment['magnetic_y'])
            key = (float(viscosity), bx, by)
            eig = cached.get(key)
            if eig is None:
                eig = spectrum(*key)
                cached[key] = eig
            active, rates, vectors, forcing = eig

            if experiment['observable'] == 'mean':
                observation = mean[active]
            elif experiment['observable'] == 'point':
                observation = (2. * np.sin(m[active]*np.pi*experiment['x']) *
                               np.sin(n[active]*np.pi*experiment['y']))
            else:
                raise ValueError("observable must be 'mean' or 'point'")

            modal_response = vectors @ (_time_factor(rates, float(experiment['time']))
                                        * forcing)
            result[index] = (float(experiment['pressure_gradient']) / DENSITY
                             * (observation @ modal_response))
        return result

    def predict(self, experiments):
        if self.viscosity is None:
            raise ValueError('fit must be called before predict')
        return np.asarray(self._predict_with_viscosity(list(experiments),
                                                        self.viscosity))
