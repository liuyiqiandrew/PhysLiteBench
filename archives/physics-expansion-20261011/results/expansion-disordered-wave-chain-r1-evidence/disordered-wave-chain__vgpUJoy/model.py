import numpy as np
from scipy.optimize import minimize_scalar


def _phase_averaged_transmission(count, wave_number, strength):
    """Return the ensemble mean transmission for one experiment.

    Write the transfer matrix of a chain in the usual SU(1, 1) form and let
    ``x = 1/T - 1``.  If a barrier is appended after a uniformly random
    separation, its phase gives

        x' = x + s + 2*x*s
             - 2*sqrt(x*s*(1+x)*(1+s))*cos(theta),

    where ``s = (strength/wave_number)**2``.  It is more numerically stable
    to use ``x = sinh(rho)**2``.  The recurrence below averages over the
    random phase with Gauss-Chebyshev quadrature and interpolates the radial
    function on a fixed grid.  The N=0, 1, and 2 cases are handled exactly.
    """
    n = int(count)
    if n == 0:
        return 1.0

    q = float(strength) / float(wave_number)
    s = q * q
    if n == 1:
        return 1.0 / (1.0 + s)
    if n == 2:
        # This is the exact uniform-phase average of the two-barrier result.
        return 1.0 / (1.0 + 2.0 * s)

    # A single barrier has radial coordinate rho=a.  After each additional
    # barrier, rho is transformed by a random relative phase.  The maximum
    # radial coordinate needed for an n-barrier chain is n*a.
    a = np.arcsinh(q)
    grid = np.linspace(0.0, n * a, 2049)
    values = 1.0 / np.cosh(grid) ** 2  # transmission at zero further steps

    # Midpoint Chebyshev nodes integrate the uniform phase without endpoint
    # singularities.  The node count and radial grid make the numerical error
    # far below the calibration noise over the stated input domain.
    nodes = (np.arange(256, dtype=float) + 0.5) * np.pi / 256.0
    phase_cos = np.cos(nodes)
    cosh_2a = np.cosh(2.0 * a)
    sinh_2a = np.sinh(2.0 * a)

    for _ in range(n - 1):
        rho = grid[:, None]
        cosh_2rho = np.cosh(2.0 * rho)
        sinh_2rho = np.sinh(2.0 * rho)
        cosh_2next = (
            cosh_2rho * cosh_2a
            - sinh_2rho * sinh_2a * phase_cos[None, :]
        )
        # Roundoff can make the argument infinitesimally less than one at
        # the rho'=0 endpoint.
        rho_next = 0.5 * np.arccosh(np.maximum(cosh_2next, 1.0))
        values = np.mean(np.interp(rho_next, grid, values), axis=1)

    return float(np.interp(a, grid, values))


def response(e, strength):
    return _phase_averaged_transmission(
        e['count'], e['wave_number'], strength
    )


def predict_at(experiments, strength):
    return np.asarray([response(e, strength) for e in experiments])


class Model:
    def __init__(self):
        self.strength = None

    def fit(self, records):
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        experiments = [record['input'] for record in records]

        def objective(strength):
            prediction = np.asarray(
                [response(experiment, strength) for experiment in experiments],
                dtype=float,
            )
            residual = (prediction - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.6, 1.0),
            method='bounded',
            options={'xatol': 1e-12},
        )
        self.strength = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.strength)
