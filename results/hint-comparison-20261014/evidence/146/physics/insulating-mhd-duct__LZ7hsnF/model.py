"""Conducting flow in an insulating rectangular duct (see README.md).

Fully developed axial flow u(x, y, t) e_z driven by a constant pressure
gradient g in a uniform transverse field B = (Bx, By, 0).  The walls are
insulating, so the induced current is divergence free and has zero normal
component: J = curl(psi e_z) with psi = 0 on all walls.  Faraday's law
(negligible magnetic Reynolds number) gives

    Laplacian(psi) = -sigma * (B . grad) u ,        psi = 0 on walls,
    (J x B)_z      = (B . grad) psi = -sigma * L G L u ,

with L = B . grad and G the Dirichlet Green operator of the Laplacian.  The
momentum balance is therefore

    rho du/dt = g + eta Laplacian(u) - sigma L G L u .

Both motional-current contributions (the Bx and By pieces) live in this single
current field, so the damping is a single symmetric operator rather than a sum
of independent axis responses.
"""

from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares

WIDTH, HEIGHT = .012, .008
DENSITY, CONDUCTIVITY = 1000., 1e5
X_MODES, Y_MODES = 40, 56
VISCOSITY_BOUNDS = (0.006, 0.018)
AREA_ROOT = np.sqrt(WIDTH*HEIGHT)


def _active_indices(nx, ny, oblique):
    """Velocity modes that can be excited.

    The constant forcing only feeds odd/odd modes, and the (symmetric) damping
    operator preserves the parity classes reachable from them.  For an oblique
    field the cross terms couple odd/odd to even/even, so both classes are
    required; for an axis-aligned field only odd/odd survives.
    """
    m, n = np.meshgrid(np.arange(1, nx+1), np.arange(1, ny+1), indexing='ij')
    m, n = m.ravel(), n.ravel()
    if oblique:
        keep = (m+n) % 2 == 0
    else:
        keep = (m % 2 == 1) & (n % 2 == 1)
    return m[keep], n[keep]


@lru_cache(4)
def _derivative_tables(nx, ny):
    """Sine-basis coefficients of d/dx and d/dy for the intermediate basis."""
    p = np.arange(1, nx+1)
    q = np.arange(1, ny+1)
    dx = np.zeros((nx, nx))
    dy = np.zeros((ny, ny))
    for target in range(nx):
        for source in range(nx):
            if (p[target]+p[source]) % 2 == 1:
                pp, mm = p[target], p[source]
                dx[target, source] = 4.*pp*mm/(WIDTH*(pp*pp-mm*mm))
    for target in range(ny):
        for source in range(ny):
            if (q[target]+q[source]) % 2 == 1:
                qq, nn = q[target], q[source]
                dy[target, source] = 4.*qq*nn/(HEIGHT*(qq*qq-nn*nn))
    wave = (p[:, None]*np.pi/WIDTH)**2+(q[None, :]*np.pi/HEIGHT)**2
    return p, q, dx, dy, 1./wave


@lru_cache(2)
def _magnetic_operator(oblique, nx=X_MODES, ny=Y_MODES):
    """Symmetric damping matrix M = D^T Lambda D in the velocity basis."""
    m, n = _active_indices(nx, ny, oblique)
    p_cut, q_cut = nx+2, ny+2
    _, _, dx, dy, inv_wave = _derivative_tables(p_cut, q_cut)
    dense = p_cut*q_cut
    columns = []
    for mm, nn in zip(m, n):
        column = np.zeros(dense)
        for pp in range(max(1, mm-1), min(p_cut, mm+1)+1):
            if (pp+mm) % 2 == 1:
                column[(pp-1)*q_cut+(nn-1)] += dx[pp-1, mm-1]
        for qq in range(max(1, nn-1), min(q_cut, nn+1)+1):
            if (qq+nn) % 2 == 1:
                column[(mm-1)*q_cut+(qq-1)] += dy[qq-1, nn-1]
        columns.append(column)
    derivative = np.array(columns).T
    operator = derivative.T @ (inv_wave.ravel()[:, None]*derivative)
    return m, n, 0.5*(operator+operator.T)


@lru_cache(64)
def _spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, magnetic = _magnetic_operator(bx != 0. and by != 0., nx, ny)
    wave2 = (m*np.pi/WIDTH)**2+(n*np.pi/HEIGHT)**2
    operator = (viscosity/DENSITY)*np.diag(wave2)+(CONDUCTIVITY/DENSITY)*magnetic
    rates, vectors = eigh(operator, check_finite=False)
    roots = np.sqrt(4./(WIDTH*HEIGHT))
    mean = np.zeros(m.size)
    odd = (m % 2 == 1) & (n % 2 == 1)
    mean[odd] = roots*4./(np.pi*np.pi*m[odd]*n[odd])
    return rates, vectors, mean, m, n


class Model:
    def __init__(self):
        self.viscosity = None

    def _response(self, viscosity, bx, by, time, observable, x=0., y=0.):
        """Velocity per unit pressure gradient for one experiment."""
        rates, vectors, mean, m, n = _spectrum(viscosity, bx, by)
        if observable == 'mean':
            source = mean
        else:
            source = (2./AREA_ROOT)*np.sin(m*np.pi*x)*np.sin(n*np.pi*y)
        forcing = vectors.T@mean
        amplitudes = vectors@(((1.-np.exp(-rates*time))/rates)*forcing)
        return source@amplitudes/DENSITY

    def _predict(self, viscosity, experiments):
        return np.array([
            self._pressure_gradient(e)*self._response(
                viscosity, float(e['magnetic_x']), float(e['magnetic_y']),
                float(e['time']), e['observable'],
                float(e.get('x', 0.)), float(e.get('y', 0.)))
            for e in experiments])

    @staticmethod
    def _pressure_gradient(experiment):
        return float(experiment['pressure_gradient'])

    def fit(self, records):
        inputs = [r['input'] for r in records]
        observed = np.array([r['value'] for r in records], float)
        sigma = np.array([r['sigma'] for r in records], float)

        def residual(viscosity):
            return (self._predict(viscosity, inputs)-observed)/sigma

        best, best_cost = None, np.inf
        for start in np.linspace(*VISCOSITY_BOUNDS, 5):
            sol = least_squares(lambda v: residual(v[0]), [start],
                                bounds=([VISCOSITY_BOUNDS[0]], [VISCOSITY_BOUNDS[1]]),
                                xtol=1e-14, ftol=1e-14, gtol=1e-14,
                                max_nfev=200)
            if sol.cost < best_cost:
                best, best_cost = sol, sol.cost
        self.viscosity = float(best.x[0])
        return self

    def predict(self, experiments):
        assert self.viscosity is not None, 'fit the model before predicting'
        return self._predict(self.viscosity, list(experiments))
