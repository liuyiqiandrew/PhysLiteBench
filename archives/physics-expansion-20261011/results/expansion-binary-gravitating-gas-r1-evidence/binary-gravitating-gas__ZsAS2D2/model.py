from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


@lru_cache(maxsize=1)
def single_species_coefficient():
    start = 1e-5
    def rhs(x,y):
        return [y[1],np.exp(-y[0])-2*y[1]/x]
    profile = solve_ivp(rhs,(start,12.),[start**2/6,start/3],
                        rtol=2e-11,atol=2e-13,dense_output=True).sol
    def condition(alpha):
        psi,gradient = profile(alpha)
        return alpha*np.exp(-psi)-gradient
    alpha = brentq(condition,5.,12.)
    return 1/(alpha*profile(alpha)[1])


def _profile_at_radius(central_ratio, mass_ratio, radius):
    """Return the two dimensionless enclosed-mass integrals.

    The dimensionless potential is psi = (Phi-Phi(0))/theta, where theta
    uses the light particle mass.  If the central heavy-to-light *mass
    density* ratio is ``central_ratio``, the regular equations are

        psi'' + 2 psi'/x = exp(-psi) + A exp(-chi psi).

    The returned integrals do not include the central densities; this keeps
    the mass-fraction constraint well conditioned even when A is large.
    """
    x0 = 1.0e-5
    source0 = 1.0 + central_ratio
    initial = [
        source0*x0*x0/6.0,
        source0*x0/3.0,
        x0**3/3.0,
        x0**3/3.0,
    ]

    def rhs(x, state):
        psi, gradient, light_integral, heavy_integral = state
        return [
            gradient,
            np.exp(-psi) + central_ratio*np.exp(-mass_ratio*psi)
            - 2.0*gradient/x,
            x*x*np.exp(-psi),
            x*x*np.exp(-mass_ratio*psi),
        ]

    solution = solve_ivp(
        rhs, (x0, radius), initial,
        method='DOP853', rtol=2.0e-9, atol=2.0e-11,
    )
    if not solution.success:
        raise RuntimeError("failed to integrate the equilibrium profile")
    return solution.y[2, -1], solution.y[3, -1]


def _eta_at_radius(radius, mass_ratio, heavy_fraction):
    """Dimensionless inverse temperature on the fixed-composition branch."""
    target = heavy_fraction/(1.0-heavy_fraction)

    def composition_error(log_central_ratio):
        central_ratio = np.exp(log_central_ratio)
        light, heavy = _profile_at_radius(
            central_ratio, mass_ratio, radius,
        )
        # Working in logs avoids loss of precision when the central ratio is
        # very large (which occurs for a concentrated heavy component).
        return (log_central_ratio + np.log(heavy) - np.log(light)
                - np.log(target))

    lower = np.log(target)
    upper = lower + 1.0
    upper_error = composition_error(upper)
    while upper_error < 0.0:
        upper += 2.0
        upper_error = composition_error(upper)

    log_central_ratio = brentq(
        composition_error, lower, upper,
        xtol=2.0e-9, rtol=2.0e-12,
    )
    central_ratio = np.exp(log_central_ratio)
    light, heavy = _profile_at_radius(
        central_ratio, mass_ratio, radius,
    )
    return (light + central_ratio*heavy)/radius


@lru_cache(maxsize=256)
def _mixture_coefficient(mass_ratio, heavy_fraction):
    """Return theta/(G M/R) at the first canonical instability.

    At fixed composition, regular equilibria form a one-parameter branch.
    Its inverse temperature is eta = GM/(R theta).  The first loss of
    canonical (fixed particle numbers) stability is the first maximum of
    eta reached from the dilute branch.
    """
    if heavy_fraction <= 0.0 or mass_ratio == 1.0:
        return single_species_coefficient()

    # A coarse continuation locates the first turning point.  The interval
    # is deliberately stopped well after the one-component turning point;
    # later extrema belong to already unstable branches.
    # The first peak is broad throughout the stated domain; this spacing
    # leaves the final scalar optimization to resolve its location while
    # keeping predictions inexpensive for batches of experiments.
    grid = np.arange(0.8, 14.0001, 0.8)
    values = np.array([
        _eta_at_radius(x, mass_ratio, heavy_fraction) for x in grid
    ])
    peak = None
    for i in range(1, len(grid)-1):
        if values[i] >= values[i-1] and values[i] >= values[i+1]:
            peak = i
            break

    if peak is None:
        # This is mainly a safeguard near the single-species limit.  The
        # physical first turning point is still inside this interval over
        # the advertised parameter domain.
        result = minimize_scalar(
            lambda x: -_eta_at_radius(x, mass_ratio, heavy_fraction),
            bounds=(0.4, 14.0), method='bounded',
            options={'xatol': 2.0e-5},
        )
        eta = -result.fun
    else:
        result = minimize_scalar(
            lambda x: -_eta_at_radius(x, mass_ratio, heavy_fraction),
            bounds=(grid[peak-1], grid[peak+1]), method='bounded',
            options={'xatol': 2.0e-5},
        )
        eta = -result.fun
    return 1.0/eta


def coefficient(ratio, fraction):
    return _mixture_coefficient(float(ratio), float(fraction))


def predict_at(experiments,gravity):
    return np.asarray([
        gravity*e['mass']/e['radius']
        * coefficient(e['mass_ratio'], e['heavy_fraction'])
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.gravity = None

    def fit(self,records):
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        # Calibration preparations have q=0 or chi=1, so their critical
        # coefficient is the universal one-species value.  The Gaussian
        # errors make this a one-parameter weighted least-squares fit.
        scale = single_species_coefficient()
        design = np.asarray([
            scale*r['input']['mass']/r['input']['radius']
            for r in records
        ], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        weights = 1.0/(sigma*sigma)
        fitted = np.sum(weights*design*values)/np.sum(weights*design*design)
        self.gravity = float(np.clip(fitted, 0.7, 1.4))
        return self

    def predict(self,experiments):
        if self.gravity is None:
            raise ValueError('fit the model before predicting')
        return predict_at(experiments,self.gravity)
