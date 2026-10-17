"""Stationary small-mass calorimetry of switched thermal contacts."""
from functools import lru_cache
import numpy as np

INPUT_FIELDS = ('force_x', 'force_y', 'drag_ratio', 'magnetic', 'temperature',
                'contrast', 'wavenumber', 'bath_contrast', 'switch_rate')


@lru_cache(None)
def spatial_grid(points):
    x = 2*np.pi*np.arange(points)/points
    modes = np.fft.fftfreq(points, 1/points)
    derivative = np.fft.ifft(1j*modes[:, None]*np.fft.fft(np.eye(points), axis=0), axis=0).real
    return x, derivative


def contact_parameters(e, friction):
    """Thermal factors, state drag, and forward (column-source) valve matrix."""
    rate = float(e['switch_rate'])
    if 'valve_cycle' not in e:
        labels = np.array([1., -1.])
        q = np.array([[-rate, rate], [rate, -rate]])
        scale = np.ones(2)
    else:
        labels = np.array([-1., 0., 1.])
        bias = float(e['valve_cycle'])
        q = np.zeros((3, 3))
        for source in range(3):
            q[(source+1) % 3, source] = rate*(1+bias)
            q[(source-1) % 3, source] = rate*(1-bias)
            q[source, source] = -2*rate
        scale = 1 + float(e.get('contact_drag_contrast', 0.))*labels
    gx = friction*scale
    return 1+float(e['bath_contrast'])*labels, gx, gx*float(e['drag_ratio']), q


def moment_operator(degree, gx, gy, magnetic, q):
    ns = len(gx)
    basis = [(p, degree-p) for p in range(degree, -1, -1)]
    index = {pair: i for i, pair in enumerate(basis)}
    a = np.kron(q, np.eye(degree+1))
    for s in range(ns):
        for i, (p, n) in enumerate(basis):
            row = s*(degree+1)+i
            a[row, row] -= p*gx[s]+n*gy[s]
            if p:
                a[row, s*(degree+1)+index[p-1, n+1]] += p*magnetic
            if n:
                a[row, s*(degree+1)+index[p+1, n-1]] -= n*magnetic
    return basis, a


def solve_moments(degree, lower, temperature, gx, gy, magnetic, q,
                  derivative=None, force=(0., 0.), prior=None, higher=None):
    """Solve a full velocity/contact cell equation at one expansion order."""
    ns, points = temperature.shape
    basis, a = moment_operator(degree, gx, gy, magnetic, q)
    source = np.zeros((ns, degree+1, points))
    for s in range(ns):
        for i, (p, n) in enumerate(basis):
            if p >= 2:
                source[s, i] += p*(p-1)*gx[s]*temperature[s]*lower[p-2, n][s]
            if n >= 2:
                source[s, i] += n*(n-1)*gy[s]*temperature[s]*lower[p, n-2][s]
            if prior is not None:
                if p:
                    source[s, i] += p*force[0]*prior[p-1, n][s]
                if n:
                    source[s, i] += n*force[1]*prior[p, n-1][s]
            if higher is not None:
                source[s, i] -= derivative @ higher[p+1, n][s]
    value = np.linalg.solve(a, -source.reshape(ns*(degree+1), points))
    return {pair: value.reshape(ns, degree+1, points)[:, i] for i, pair in enumerate(basis)}


def stationary_contact(q):
    qa = q.copy()
    qa[-1] = 1.
    rhs = np.zeros(len(q)); rhs[-1] = 1.
    return np.linalg.solve(qa, rhs)


def stationary_density(temperatures, gx, gy, magnetic, q, force, derivative):
    """Positional equation from the velocity cell problem, including state drag."""
    ns, points = temperatures.shape
    _, a1 = moment_operator(1, gx, gy, magnetic, q)
    _, a2 = moment_operator(2, gx, gy, magnetic, q)
    inv1, inv2 = np.linalg.inv(a1), np.linalg.inv(a2)
    pi = stationary_contact(q)
    # M2^0 = K2 [T_s rho]_s.
    k2 = np.zeros((3*ns, ns))
    for s in range(ns):
        source = np.zeros(3*ns)
        source[3*s] = 2*gx[s]*pi[s]
        source[3*s+2] = 2*gy[s]*pi[s]
        k2[:, s] = -inv2 @ source
    sx = np.zeros(2*ns); sx[0::2] = 1.
    # Streaming maps (u_x^2,u_xu_y,u_y^2) to (u_x,u_y).
    stream = np.zeros((2*ns, 3*ns))
    for s in range(ns):
        stream[2*s, 3*s] = 1.
        stream[2*s+1, 3*s+1] = 1.
    c = sx @ inv1 @ stream @ k2
    force_source = np.zeros(2*ns)
    force_source[0::2] = force[0]*pi
    force_source[1::2] = force[1]*pi
    force_coefficient = -sx @ inv1 @ force_source
    flux = force_coefficient*np.eye(points)
    for s in range(ns):
        flux += c[s] * derivative @ np.diag(temperatures[s])
    equation = derivative @ flux
    dx = 2*np.pi/points
    equation[-1] = dx
    rhs = np.zeros(points); rhs[-1] = 1.
    return np.linalg.solve(equation, rhs), pi


def conditional_density_correction(q, derivative, first):
    """Q M0^(2)=d_x M1^(1), in zero-total-density gauge.

    The null component cancels exactly against the leading background rate
    times the measured residence.  The contact-conditioned component does
    not, particularly for a directed cycle.
    """
    ns, points = first[1, 0].shape
    qa = q.copy(); qa[-1] = 1.
    rhs = derivative @ first[1, 0].T
    out = np.empty((ns, points))
    for j in range(points):
        b = rhs[j].copy(); b[-1] = 0.
        out[:, j] = np.linalg.solve(qa, b)
    return {(0, 0): out}


def limiting_state(e, friction, points=129):
    phase, derivative = spatial_grid(points)
    derivative = int(e['wavenumber'])*derivative
    dx = 2*np.pi/points
    factors, gx, gy, q = contact_parameters(e, friction)
    t0 = float(e['temperature'])*(1+float(e['contrast'])*np.cos(phase))
    temperatures = factors[:, None]*t0
    force = (float(e['force_x']), float(e['force_y']))
    magnetic = float(e['magnetic'])
    density, pi = stationary_density(temperatures, gx, gy, magnetic, q, force, derivative)
    zeroth = {(0, 0): pi[:, None]*density}

    # With u=sqrt(m)v: M2^0, M4^0, M1^1, M0^2, M3^1, M2^2.
    second = solve_moments(2, zeroth, temperatures, gx, gy, magnetic, q)
    fourth = solve_moments(4, second, temperatures, gx, gy, magnetic, q)
    first = solve_moments(1, {}, temperatures, gx, gy, magnetic, q,
                          derivative, force, zeroth, second)
    density_correction = conditional_density_correction(q, derivative, first)
    third = solve_moments(3, first, temperatures, gx, gy, magnetic, q,
                          derivative, force, second, fourth)
    second_correction = solve_moments(2, density_correction, temperatures, gx, gy,
                                      magnetic, q, derivative, force, first, third)
    local = np.sum(
        gx[:, None]*(second_correction[2, 0]/temperatures-density_correction[0, 0])
        + gy[:, None]*(second_correction[0, 2]/temperatures-density_correction[0, 0]), axis=0)
    return dict(value=float(dx*local.sum()), local=local, phase=phase, density=density,
                dx=dx, first=first, second=second, temperatures=temperatures,
                density_correction=density_correction, second_correction=second_correction)


@lru_cache(4096)
def entropy_rate(force_x, force_y, drag_ratio, magnetic, temperature, contrast,
                 wavenumber, bath_contrast, switch_rate, friction,
                 detector_phase=None, valve_cycle=None, contact_drag_contrast=0.):
    e = dict(zip(INPUT_FIELDS, (force_x, force_y, drag_ratio, magnetic, temperature,
                                contrast, wavenumber, bath_contrast, switch_rate)))
    if valve_cycle is not None:
        e.update(valve_cycle=valve_cycle, contact_drag_contrast=contact_drag_contrast)
    state = limiting_state(e, friction)
    weight = 1. if detector_phase is None else (1+np.cos(state['phase']-detector_phase))/2
    return float(state['dx']*np.sum(weight*state['local']))


def coefficients(experiments, friction=1.):
    return np.asarray([entropy_rate(*(e[f] for f in INPUT_FIELDS), float(friction),
                                    e.get('detector_phase'), e.get('valve_cycle'),
                                    e.get('contact_drag_contrast', 0.))
                       for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        inputs = [r['input'] for r in records]

        def objective(friction):
            residual = (coefficients(inputs, friction)-values)/sigma
            return float(residual @ residual)

        # Bounded golden-section minimization avoids depending on scipy.
        a, b = .7, 1.6
        ratio = (np.sqrt(5)-1)/2
        c, d = b-ratio*(b-a), a+ratio*(b-a)
        fc, fd = objective(c), objective(d)
        for _ in range(18):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b-ratio*(b-a); fc = objective(c)
            else:
                a, c, fc = c, d, fd
                d = a+ratio*(b-a); fd = objective(d)
        self.friction = float((a+b)/2)
        return self

    def predict(self, experiments):
        if self.friction is None:
            raise RuntimeError('Call fit(records) before predict(experiments).')
        return coefficients(experiments, self.friction)
