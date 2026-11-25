"""Stationary heat cumulants for the retained, finite gas contact.

The backward generator is projected onto Gaussian Hermite polynomials in
(v+g, v-g). In these coordinates a collision simply reverses the second
coordinate. Its absolute-speed rate can therefore be integrated exactly,
without smoothing the cusp or replacing the retained gas by fresh draws.
Fourier modes retain the physical detector angle in the heat cumulants.
"""
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import splu
from scipy.special import roots_laguerre

RATE = 3.6
FIELDS = ('angular_speed', 'flip_rate', 'temperature_a', 'temperature_b',
          'flux_fraction', 'relaxation_rate')
STATISTICS = ('mean', 'noise', 'third_cumulant')


def calibration_mean(e, drag):
    """Exact heat conductance for the constant collision clock at zero flow."""
    lam = e['relaxation_rate']
    return (drag * lam * RATE / (2 * drag * lam + RATE * (drag + lam))
            * (e['temperature_a'] - e['temperature_b']))


def fit_drag(records):
    """Weighted least squares in the exactly known calibration conductance."""
    records = list(records)
    lam = float(records[0]['input']['relaxation_rate'])
    contrast = np.array([(r['input']['temperature_a'] - r['input']['temperature_b'])
                         / r['sigma'] for r in records])
    measured = np.array([r['value'] / r['sigma'] for r in records])
    conductance = float(contrast @ measured / (contrast @ contrast))
    return float(np.clip(conductance * lam * RATE
                         / (lam * RATE - conductance * (2 * lam + RATE)), .4, 1.1))


def normalized_hermites(z, degree):
    z = np.asarray(z)
    h = np.empty((z.size, degree + 1))
    h[:, 0] = 1.
    if degree:
        h[:, 1] = z
    for n in range(1, degree):
        h[:, n + 1] = (z * h[:, n] - np.sqrt(n) * h[:, n - 1]) / np.sqrt(n + 1)
    return h


@lru_cache(maxsize=8)
def _basis(degree):
    """Parameter-independent sparse operators in a total-degree basis.

    x=(v+g)/sqrt(2*variance), y=(v-g)/sqrt(2*variance).
    Columns hold the coefficients of the operator acting on a basis function.
    """
    pairs = [(n, t - n) for t in range(degree + 1) for n in range(t + 1)]
    lookup = {pair: j for j, pair in enumerate(pairs)}
    size = len(pairs)

    def matrix(terms):
        rows, cols, values = [], [], []
        for j, (n, m) in enumerate(pairs):
            for a, b, value in terms(n, m):
                i = lookup.get((a, b))
                if i is not None and value != 0:
                    rows.append(i)
                    cols.append(j)
                    values.append(value)
        return coo_matrix((values, (rows, cols)), shape=(size, size)).tocsc()

    dx = matrix(lambda n, m: [(n - 1, m, np.sqrt(n))])
    dy = matrix(lambda n, m: [(n, m - 1, np.sqrt(m))])
    x = matrix(lambda n, m: [(n - 1, m, np.sqrt(n)), (n + 1, m, np.sqrt(n + 1))])
    y = matrix(lambda n, m: [(n, m - 1, np.sqrt(m)), (n, m + 1, np.sqrt(m + 1))])
    # Direct multiplication by (x-y)^2 avoids a projection at the intermediate
    # degree: squaring a truncated multiplication matrix would lose edge terms.
    relative_square = matrix(lambda n, m: [
        (n, m, 2 * (n + m + 1)),
        (n - 2, m, np.sqrt(n * max(n - 1, 0))),
        (n + 2, m, np.sqrt((n + 1) * (n + 2))),
        (n, m - 2, np.sqrt(m * max(m - 1, 0))),
        (n, m + 2, np.sqrt((m + 1) * (m + 2))),
        (n - 1, m - 1, -2 * np.sqrt(n * m)),
        (n - 1, m + 1, -2 * np.sqrt(n * (m + 1))),
        (n + 1, m - 1, -2 * np.sqrt((n + 1) * m)),
        (n + 1, m + 1, -2 * np.sqrt((n + 1) * (m + 1))),
    ])
    # On odd y polynomials the exchange difference is -2. With t=y^2/2,
    # the |y| Gaussian integral is an ordinary Laguerre polynomial integral.
    nodes, weights = roots_laguerre(degree + 2)
    odd = np.arange(1, degree + 1, 2)
    h = normalized_hermites(np.sqrt(2 * nodes), degree)[:, odd]
    flux_block = -4 / np.sqrt(np.pi) * (h.T @ (weights[:, None] * h))
    rows, cols, values = [], [], []
    for n in range(degree):
        ms = odd[odd <= degree - n]
        ids = [lookup[(n, int(m))] for m in ms]
        for a, i in enumerate(ids):
            for b, j in enumerate(ids):
                rows.append(i)
                cols.append(j)
                values.append(flux_block[a, b])
    flux = coo_matrix((values, (rows, cols)), shape=(size, size)).tocsc()
    exchange = diags([-2. if m % 2 else 0. for n, m in pairs], format='csc')
    parity = diags([(-1.) ** (n + m) - 1 for n, m in pairs], format='csc')
    return dx, dy, x, y, relative_square, flux, exchange, parity


def _operators(e, drag, degree):
    ta, tb = e['temperature_a'], e['temperature_b']
    lam, eta, omega = e['relaxation_rate'], e['flux_fraction'], e['angular_speed']
    variance = max(ta, tb)
    dx, dy, x, y, square, flux, exchange, parity = _basis(degree)
    identity = eye(x.shape[0], format='csc')
    a, b = (drag + lam) / 2, (drag - lam) / 2
    d = (drag * ta + lam * tb) / (2 * variance)
    f = (drag * ta - lam * tb) / (2 * variance)
    L = (-a * (x @ dx + y @ dy) - b * (y @ dx + x @ dy)
         + d * (dx @ dx + dy @ dy) + 2 * f * (dx @ dy)
         + RATE * ((1 - eta) * exchange + eta * np.sqrt(variance) * flux))
    if omega:
        # Work in (s*v, s*g): the target is fixed and a motor flip reverses
        # both coordinates. Heat is invariant under that simultaneous reversal.
        L = (L + lam * omega / np.sqrt(2 * variance) * (dx - dy)
             + e['flip_rate'] * parity)
    G = np.sqrt(variance / 2) * (x - y)
    V = np.sqrt(variance / 2) * (x + y)
    dg = (dx - dy) / np.sqrt(2 * variance)
    relative_g = G - omega * identity
    relative_g2 = variance / 2 * square - 2 * omega * G + omega ** 2 * identity
    # L(k)=L+k*B+k^2*C/2 for E[exp(k*Q)]. The derivative in B is the
    # cross variation of Q with g, using the SAME bath-B Wiener process.
    B = lam * (relative_g2 - tb * identity) - 2 * lam * tb * (relative_g @ dg)
    C = 2 * lam * tb * relative_g2
    return L.tocsc(), V, B.tocsc(), C.tocsc()


def _cumulants(e, drag, degree):
    L, V, B, C = _operators(e, drag, degree)
    size = L.shape[0]
    # L annihilates the constant polynomial. The remaining block determines
    # both the stationary moments and the centered Poisson solutions.
    factor = splu(L[1:, 1:].tocsc())
    p = np.empty(size)
    p[0] = 1.
    p[1:] = factor.solve(-L[0, 1:].toarray().ravel(), trans='T')

    def invert_zero(rhs):
        result = np.empty(size)
        result[1:] = factor.solve(np.asarray(rhs[1:]))
        result[0] = -p[1:] @ result[1:]
        return result

    b0 = B[:, 0].toarray().ravel()
    c0 = C[:, 0].toarray().ravel()
    mean = float(p @ b0)
    r10 = invert_zero(-b0)
    eps = e.get('detector_contrast', 0.)
    if not eps:
        noise = float(p @ (c0 + 2 * (B @ r10)))
        r20 = invert_zero(2 * mean * r10 - 2 * (B @ r10) - c0)
        third = float(3 * p @ (B @ r20 + C @ r10))
    else:
        # Multiplication by w couples Fourier modes 0,+/-h; w^2 also has
        # +/-2h. Through order three only responses at 0,+/-h can return
        # to mode zero, so no angular discretization or mixing closure is used.
        harmonic = e.get('detector_harmonic', 1)
        angular_factor = splu((L + 1j * harmonic * V).tocsc())
        r11 = angular_factor.solve(-eps / 2 * b0)
        averaged_r1 = r10 + eps * r11.real
        noise = float(p @ ((1 + eps ** 2 / 2) * c0 + 2 * (B @ averaged_r1)))
        r20 = invert_zero(2 * mean * r10 - 2 * (B @ averaged_r1)
                          - (1 + eps ** 2 / 2) * c0)
        r21 = angular_factor.solve(2 * mean * r11
                                  - 2 * (B @ (r11 + eps / 2 * r10)) - eps * c0)
        third = float(3 * p @ (B @ (r20 + eps * r21.real)
                              + C @ ((1 + eps ** 2 / 2) * r10 + 2 * eps * r11.real)))
    return dict(mean=mean, noise=noise, third_cumulant=third)


@lru_cache(maxsize=512)
def _readouts(angular_speed, flip_rate, temperature_a, temperature_b,
              flux_fraction, relaxation_rate, contrast, harmonic, drag):
    e = dict(zip(FIELDS, (angular_speed, flip_rate, temperature_a, temperature_b,
                         flux_fraction, relaxation_rate)))
    e.update(detector_contrast=contrast, detector_harmonic=harmonic)
    # With a constant clock and no angular weighting, degree six closes
    # exactly for the first three heat cumulants. The absolute-speed clock
    # and the angular resolvent require a converged spectral expansion.
    if flux_fraction == 0:
        degree = 6 if contrast == 0 else 64
    else:
        degree = 96
    return _cumulants(e, drag, degree)


def predict_at(experiments, drag):
    result = []
    for e in experiments:
        statistic = e.get('statistic', 'mean')
        if statistic not in STATISTICS:
            raise ValueError('Unknown calorimeter statistic: ' + str(statistic))
        contrast = float(e.get('detector_contrast', 0.))
        harmonic = int(e.get('detector_harmonic', 1))
        if contrast and e['angular_speed'] != 0:
            raise ValueError('Angularly gated preparations require zero flow target.')
        if statistic == 'mean' and e['flux_fraction'] == 0 and e['angular_speed'] == 0:
            value = calibration_mean(e, drag)
        else:
            # The uniform stationary angle makes the mean independent of gain.
            values = _readouts(*(float(e[f]) for f in FIELDS),
                               contrast if statistic != 'mean' else 0., harmonic, float(drag))
            value = values[statistic]
        result.append(value)
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.drag)
