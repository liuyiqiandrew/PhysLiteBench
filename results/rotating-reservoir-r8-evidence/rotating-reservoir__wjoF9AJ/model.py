"""Model for the finite rotor--gas contact.

The zero-flux calibration problem has an exact closed form.  Away from the
calibration surface the continuous state is not Gaussian (the collision clock
depends on the relative speed), so the implementation below uses a small
conditional-Gaussian moment closure.  The motor label is kept as an explicit
two-state variable.  This is important: replacing ``s * angular_speed`` by a
constant angular speed gives the wrong answer even when the label has equal
stationary probabilities.
"""
from functools import lru_cache
from math import erf
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

RATE=3.6

def folded(mu, variance):
    sigma=np.sqrt(max(variance,1e-14));gaussian=np.sqrt(2/np.pi)*np.exp(-mu*mu/(2*sigma*sigma))
    sign=erf(mu/(np.sqrt(2)*sigma))
    absolute=sigma*gaussian+mu*sign
    signed_square=(mu*mu+sigma*sigma)*sign+mu*sigma*gaussian
    absolute_cube=(mu**3+3*mu*sigma*sigma)*sign+(mu*mu+2*sigma*sigma)*sigma*gaussian
    return absolute,signed_square,absolute_cube


def _stationary_moments(e, gamma):
    """Return stationary moments of (v, g, s*v, s*g).

    Unconditional first moments of v and g vanish by the sign symmetry of the
    apparatus.  The two label-weighted first moments do not vanish and carry
    the response to the moving bath.  The relative-speed part of the collision
    clock is closed with Gaussian absolute moments.
    """
    ta = float(e['temperature_a'])
    tb = float(e['temperature_b'])
    omega = float(e['angular_speed'])
    lam = float(e['relaxation_rate'])
    eta = float(e['flux_fraction'])
    flip = float(e['flip_rate'])

    def derivative(_t, y):
        k, h, vv, gg, vg = y
        dvar = max(vv + gg - 2.0 * vg, 1.0e-14)
        # E[|d| d^2] / E[d^2], for a centered Gaussian d.
        abs_factor = 2.0 * np.sqrt(2.0 / np.pi) * np.sqrt(dvar)

        # Conditional on s, d has mean s*(k-h).  This gives the more useful
        # folded moment for the label-weighted collision terms.
        cdvar = max(dvar - (k - h) ** 2, 1.0e-14)
        label_flux = (1.0 - eta) * (k - h) + eta * folded(k - h, cdvar)[1]
        pair_flux = (1.0 - eta) + eta * abs_factor
        exchange = RATE * pair_flux * (gg - vv)

        return np.array([
            -gamma * k - RATE * label_flux - 2.0 * flip * k,
            -lam * (h - omega) + RATE * label_flux - 2.0 * flip * h,
            -2.0 * gamma * (vv - ta) + exchange,
            -2.0 * lam * (gg - omega * h - tb) - exchange,
            -(gamma + lam) * vg + lam * omega * k,
        ])

    initial = np.array([0.0, 0.0, ta, tb, 0.0])
    solution = solve_ivp(
        derivative, (0.0, 260.0), initial, method='BDF',
        rtol=2e-9, atol=2e-11,
    )
    if not solution.success:
        raise RuntimeError('stationary moment integration failed')
    k, h, vv, gg, vg = solution.y[:, -1]
    # Small numerical negative eigenvalues can occur at the end of a stiff
    # solve.  Keep the closure usable at the edges of the documented range.
    covariance = np.array([[vv, vg], [vg, gg]], dtype=float)
    eig = np.linalg.eigvalsh(covariance)
    if eig[0] < 0.0:
        covariance += np.eye(2) * (-eig[0] + 1e-12)
        vv, vg, gg = covariance[0, 0], covariance[0, 1], covariance[1, 1]
    return np.array([k, h, vv, gg, vg], dtype=float)


def _quadratic_response(e, gamma, moments, mean):
    """Project the generator and the heat Poisson equation onto quadratics."""
    ta = float(e['temperature_a'])
    tb = float(e['temperature_b'])
    omega = float(e['angular_speed'])
    lam = float(e['relaxation_rate'])
    eta = float(e['flux_fraction'])
    flip = float(e['flip_rate'])
    k, h, vv, gg, vg = moments

    # Approximate the conditional law of (v,g) given the motor label by a
    # Gaussian.  Its conditional covariance follows from E[s v] and E[s g].
    conditional = np.array([[vv - k*k, vg - k*h], [vg - k*h, gg - h*h]])
    values, vectors = np.linalg.eigh(conditional)
    values = np.maximum(values, 1e-12)
    transform = vectors @ np.diag(np.sqrt(values))
    nodes, weights = hermgauss(9)
    z1, z2 = np.meshgrid(nodes, nodes, indexing='ij')
    qw = (weights[:, None] * weights[None, :] / np.pi).ravel() * 0.5
    z = np.column_stack((z1.ravel(), z2.ravel())) * np.sqrt(2.0)

    all_v = []
    all_g = []
    all_s = []
    all_w = []
    for label in (-1.0, 1.0):
        x = label * np.array([k, h]) + z @ transform.T
        all_v.append(x[:, 0])
        all_g.append(x[:, 1])
        all_s.append(np.full(len(x), label))
        all_w.append(qw)
    v = np.concatenate(all_v)
    g = np.concatenate(all_g)
    s = np.concatenate(all_s)
    w = np.concatenate(all_w)

    # b = (1, v, g, s, v^2, vg, g^2, sv, sg).
    b = np.column_stack((
        np.ones_like(v), v, g, s, v*v, v*g, g*g, s*v, s*g,
    ))
    dv = -gamma * v
    dg = -lam * (g - s * omega)
    collision = RATE * ((1.0 - eta) + eta * np.abs(v - g))
    swapped = np.column_stack((
        np.ones_like(v), g, v, s, g*g, v*g, v*v, s*g, s*v,
    ))
    local = np.column_stack((
        np.zeros_like(v),
        dv,
        dg,
        -2.0 * flip * s,
        2.0 * v * dv + 2.0 * gamma * ta,
        dv * g + v * dg,
        2.0 * g * dg + 2.0 * lam * tb,
        s * dv - 2.0 * flip * s * v,
        s * dg - 2.0 * flip * s * g,
    ))
    local += collision[:, None] * (swapped - b)

    gram = b.T @ (w[:, None] * b)
    generator = np.linalg.solve(gram, b.T @ (w[:, None] * local))
    stationary = np.array([1.0, 0.0, 0.0, 0.0, vv, vg, gg, k, h])

    # Heat drift f = a dot b.  Solve L phi = mean - f, with E[phi]=0.
    a = np.array([
        lam * (omega * omega - tb), 0.0, 0.0, 0.0,
        0.0, 0.0, lam, 0.0, -2.0 * lam * omega,
    ])
    rhs = mean * np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) - a
    system = generator.copy()
    system[0, :] = stationary
    rhs[0] = 0.0
    phi = np.linalg.solve(system, rhs)

    gradient_g = phi[2] + phi[5] * v + 2.0 * phi[6] * g + phi[8] * s
    heat_noise = -np.sqrt(2.0 * lam * tb) * (g - s * omega)
    particle_noise = np.sqrt(2.0 * lam * tb) * gradient_g
    shot = float(w @ (heat_noise * heat_noise))
    correlation = 2.0 * float(a @ gram @ phi + w @ (heat_noise * particle_noise))
    return max(0.0, shot + correlation)


def gaussian_closure(e, gamma, noise=False):
    moments = _stationary_moments(e, gamma)
    k, h, vv, gg, vg = moments
    omega = float(e['angular_speed'])
    lam = float(e['relaxation_rate'])
    tb = float(e['temperature_b'])
    mean = lam * (gg - 2.0 * omega * h + omega * omega - tb)
    result = dict(mean=float(mean), vmean=0.0, gmean=0.0,
                  v2=float(vv), g2=float(gg), vg=float(vg),
                  motor_v=float(k), motor_g=float(h))
    if noise:
        result['noise'] = float(_quadratic_response(e, gamma, moments, mean))
    return result


@lru_cache(4096)
def readout(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction,relaxation_rate,statistic,drag):
    e=dict(zip(FIELDS,(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction,relaxation_rate)))
    if statistic not in ['mean','noise']:raise ValueError('statistic must be mean or noise')
    if statistic=='mean' and flux_fraction==0. and angular_speed==0.:
        return float(calibration_mean(e,drag))
    return gaussian_closure(e,drag,noise=statistic=='noise')[statistic]


RATE=3.6
FIELDS=('angular_speed','flip_rate','temperature_a','temperature_b','flux_fraction','relaxation_rate')


def calibration_mean(e,drag):
    lam=e['relaxation_rate']
    return drag*lam*RATE/(2*drag*lam+RATE*(drag+lam))*(e['temperature_a']-e['temperature_b'])


def fit_drag(records):
    if not records:
        raise ValueError('at least one calibration record is required')

    def prediction(record, drag):
        item = record['input']
        return calibration_mean(item, drag)

    values = np.asarray([float(r['value']) for r in records])
    sigmas = np.asarray([float(r.get('sigma', 1.0)) for r in records])
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError('calibration values and uncertainties must be finite')

    def objective(drag):
        expected = np.asarray([prediction(r, drag) for r in records])
        residual = (expected - values) / sigmas
        return float(residual @ residual)

    result = minimize_scalar(objective, bounds=(0.4, 1.1), method='bounded',
                             options={'xatol': 1e-12})
    if not result.success:
        raise RuntimeError('drag calibration failed')
    return float(np.clip(result.x, 0.4, 1.1))


def predict_at(experiments,drag):
    return np.array([readout(*(float(e[f]) for f in FIELDS),e.get('statistic','mean'),float(drag)) for e in experiments])


class Model:
    def __init__(self):self.drag=None

    def fit(self,records):
        self.drag = fit_drag(records)
        # The fit is commonly followed by predictions for the same settings;
        # clearing this cache also prevents a previous model fit from leaking
        # into a newly calibrated instance in long-running processes.
        readout.cache_clear()
        return self

    def predict(self,experiments):
        if self.drag is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments,self.drag)
