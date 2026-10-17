"""Long-time center-current statistics for the annealed rod preparation.

Removing the occupied lengths gives a Poisson gas of density
``c = density / (1 - density * length)``. For a positive current j, the
observation point in this gas is displaced by the sum of j rod lengths.
The lengths stay attached to the ordered rods, independently of the free
gas. Averaging over that sum uses the length Laplace transform

    E[exp(-k A)] = exp(-k * length) * cosh(k * length * contrast).

Consequently, for lambda = abs(counting_field), the nonnegative saddle s
satisfies ``lambda = s - log(E[exp(-c * A * sinh(s))])`` and the SCGF is
``speed * c * (cosh(s) - 1)``. The contrast correction starts at order
abs(lambda)**3: the variance growth still equals speed*rho*(1-rho*length).
"""

import numpy as np
from scipy.optimize import brentq


def response(e):
    """Return the requested growth rate at unit speed."""
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    if e['statistic'] != 'scgf':
        raise ValueError(f"Unknown statistic: {e['statistic']!r}")

    field = abs(e['counting_field'])
    if field == 0.:
        return 0.
    c = rho/(1-rho*mean)
    contrast = e['contrast']

    def saddle_equation(fraction):
        # Solve for s/field so the root tolerance also works near zero field.
        excluded = c*mean*np.sinh(field*fraction)
        z = contrast*excluded
        # log(cosh(z)), without subtracting nearly equal numbers near zero.
        # The stated input domain keeps the sinh arguments safely bounded.
        log_cosh = np.log1p(2*np.sinh(z/2)**2)
        return fraction + (excluded-log_cosh)/field - 1.

    # The derivative in s is 1+c*mean*cosh(s)*(1-contrast*tanh(z)) > 0.
    # Hence there is a unique root between zero and field.
    s = field*brentq(saddle_equation, 0., 1., xtol=5e-15, rtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the shared speed by bounded Gaussian maximum likelihood."""
        records = list(records)
        unit = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0.)):
            raise ValueError('Calibration values must be finite and sigma positive and finite.')

        # All rates scale linearly with speed, so weighted least squares has
        # an exact solution; clipping gives the constrained optimum.
        design, target = unit/sigma, values/sigma
        information = np.dot(design, design)
        if information == 0.:
            raise ValueError('Calibration must contain a nonzero speed response.')
        self.speed = float(np.clip(np.dot(design, target)/information, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise ValueError('Fit the model before predicting.')
        return predict_at(experiments, self.speed)
