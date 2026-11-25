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


def coefficient(ratio,fraction):
    """Return the inverse critical dimensionless inverse temperature.

    For unequal species this is obtained from the regular two-component
    isothermal-sphere sequence.  ``fraction`` is a mass fraction, so the
    central-density ratio used by the dimensionless equations is not the
    particle-number fraction.
    """
    ratio = float(ratio)
    fraction = float(fraction)
    if fraction <= 0.0 or np.isclose(ratio, 1.0, rtol=0.0, atol=1.e-12):
        return single_species_coefficient()
    return _multicomponent_coefficient(ratio, fraction)


def _two_species_state(radius, ratio, central_ratio):
    """Integrate one dimensionless two-species equilibrium profile.

    The light central density sets the length scale and ``central_ratio`` is
    chi*n_h(0)/n_l(0).  The returned values are the enclosed heavy mass
    fraction and eta = G*M/(R*theta), respectively.
    """
    start = 1.e-5
    central_source = 1.0 + central_ratio

    def rhs(x, y):
        potential, gradient, light_integral, heavy_integral = y
        light = np.exp(-potential)
        heavy = np.exp(-ratio * potential)
        return [
            gradient,
            light + central_ratio * heavy - 2.0 * gradient / x,
            x * x * light,
            x * x * heavy,
        ]

    initial = [
        central_source * start * start / 6.0,
        central_source * start / 3.0,
        start**3 / 3.0,
        start**3 / 3.0,
    ]
    solution = solve_ivp(
        rhs,
        (start, radius),
        initial,
        rtol=2.e-8,
        atol=2.e-10,
    )
    potential, gradient, light_integral, heavy_integral = solution.y[:, -1]
    heavy_mass_fraction = (
        central_ratio * heavy_integral /
        (light_integral + central_ratio * heavy_integral)
    )
    return heavy_mass_fraction, radius * gradient


def _central_ratio(radius, ratio, fraction):
    """Find the dilute-connected central-density ratio at a given radius."""
    target = fraction / (1.0 - fraction)

    def residual(log_ratio):
        central_ratio = np.exp(log_ratio)
        actual, _ = _two_species_state(radius, ratio, central_ratio)
        return actual - fraction

    # Along the branch issuing from the dilute state the central heavy
    # density ratio grows from target.  Doubling brackets its first crossing,
    # before the (irrelevant here) high-density secondary branch can occur.
    low = np.log(target) - 2.0
    low_value = residual(low)
    high = np.log(target) + 1.0
    high_value = residual(high)
    while high_value < 0.0 and high < 45.0:
        high += np.log(2.0)
        high_value = residual(high)
    if low_value * high_value > 0.0:
        # This fallback scans for the first sign change if the upper end has
        # passed a high-density turn of the mass-fraction curve.
        grid = np.linspace(low, 45.0, 80)
        previous = low_value
        previous_x = low
        for current_x in grid[1:]:
            current = residual(current_x)
            if previous * current <= 0.0:
                low, high = previous_x, current_x
                break
            previous, previous_x = current, current_x
        else:
            raise RuntimeError("Could not locate the two-species equilibrium branch")
    return np.exp(brentq(residual, low, high, xtol=2.e-7, rtol=2.e-10))


@lru_cache(maxsize=256)
def _multicomponent_coefficient(ratio, fraction):
    """Compute the first turning point of the fixed-composition branch."""
    # The dilute branch starts at eta=0.  Find its first local maximum as the
    # dimensionless radius is increased; that is the first canonical
    # (fixed particle numbers) instability.
    previous_radius = None
    previous_eta = None
    current_radius = 0.5
    step = 0.5
    while current_radius <= 16.0:
        central_ratio = _central_ratio(current_radius, ratio, fraction)
        current_eta = _two_species_state(
            current_radius, ratio, central_ratio
        )[1]
        if previous_eta is not None and current_eta < previous_eta:
            left = previous_radius - step
            right = current_radius

            def negative_eta(radius):
                central_ratio = _central_ratio(radius, ratio, fraction)
                return -_two_species_state(radius, ratio, central_ratio)[1]

            result = minimize_scalar(
                negative_eta,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 5.e-7},
            )
            return 1.0 / (-result.fun)
        previous_radius = current_radius
        previous_eta = current_eta
        current_radius += step

    raise RuntimeError("Could not locate the first stability turning point")


def predict_at(experiments,gravity):
    return np.asarray([gravity*e['mass']/e['radius']*
                       coefficient(e['mass_ratio'],e['heavy_fraction']) for e in experiments])


class Model:
    def __init__(self):
        self.gravity = None

    def fit(self,records):
        x = []
        y = []
        weights = []
        single = single_species_coefficient()
        for record in records:
            experiment = record['input']
            x.append(experiment['mass'] / experiment['radius'] * single)
            y.append(record['value'])
            weights.append(1.0 / record['sigma']**2)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        weights = np.asarray(weights, dtype=float)
        fitted = np.sum(weights * x * y) / np.sum(weights * x * x)
        self.gravity = float(np.clip(fitted, 0.7, 1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.gravity)
