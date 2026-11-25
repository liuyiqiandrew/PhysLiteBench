import numpy as np
from scipy.optimize import brentq

MU = 1e6
DENSITY_MIN = 900.0
DENSITY_MAX = 1300.0


def geometry(e):
    s = e['stretch']
    pressure = e['pressure'] / MU
    t = brentq(lambda t: t*t - 1 + 2*np.log(s*t) + pressure*s*t, .2, 2.)
    return s, t


def surface_matrix(x, s, t):
    """Return the legacy two-column traction matrix for a mode parameter.

    This helper is kept for compatibility with the original module.  The
    pressure-dependent prediction uses ``_secular`` below, which includes the
    incremental traction from the follower pressure load.
    """
    jacobian = s*t
    c = 1 - 2*np.log(jacobian)
    b = t*t
    longitudinal = np.sqrt(1 - b*x/(b+2+c))
    transverse = np.sqrt(1-x)
    decay = np.array([longitudinal, transverse])
    tangent = np.array([1., transverse])
    normal = np.array([-longitudinal, -1.])
    shear_traction = -b*decay*tangent + c*normal
    normal_traction = -(b+2+c)*decay*normal - 2*tangent
    return np.array([shear_traction, normal_traction])


def mode(e):
    """Return the dimensionless squared wave speed for one experiment.

    ``y = density * v**2 / MU`` is the squared speed in reference-density
    units.  The two attenuation roots are obtained from the incremental bulk
    equations, and the determinant below imposes zero incremental traction on
    the pressured face.  The pressure terms in that determinant are required
    because the actuator maintains pressure per *current* area.
    """
    s, t = geometry(e)
    pressure = e['pressure'] / MU
    jacobian = s*t
    pressure_load = pressure*jacobian
    b = t*t
    c = 1 - 2*np.log(jacobian)
    bulk_xx = s*s + c + 2
    bulk_zz = b + c + 2
    coupling = (c + 2)**2

    def secular(y):
        # With u ~ exp(i k x - k q z), the characteristic equation is a
        # quadratic in Q=q**2.  Its roots are positive throughout the stated
        # stable, subsonic input range.
        first = bulk_xx - y
        second = s*s - y
        coefficients = np.array([
            b*bulk_zz,
            -first*bulk_zz - b*second + coupling,
            first*second,
        ])
        attenuation_squared = np.sort(np.roots(coefficients))

        traction_columns = []
        for q_squared in attenuation_squared:
            q = np.sqrt(q_squared)
            acoustic = bulk_xx - b*q_squared - y

            # An eigenvector can be chosen as
            # (i*q*(c+2), -acoustic).  Its incremental tractions are the two
            # entries in this column.  The imaginary part is a real secular
            # function after the two roots are ordered.
            traction_columns.append([
                -1j * (b*q_squared*(c + 2)
                        + (c - pressure_load)*acoustic),
                q * (bulk_zz*acoustic
                     - (2 + pressure_load)*(c + 2)),
            ])
        return np.linalg.det(np.asarray(traction_columns).T).imag

    y = brentq(
        secular,
        1e-8*s*s,
        s*s*(1 - 1e-10),
        xtol=1e-13,
    )
    return y


def predict_at(experiments, density):
    return np.sqrt(MU*np.array([mode(e) for e in experiments])/density)


class Model:
    def __init__(self):
        self.density = None

    def fit(self, records):
        """Estimate the (reference) density from calibration measurements.

        For a fixed deformation the secular equation determines a factor ``a``
        such that the predicted speed is ``a / sqrt(density)``.  Consequently
        the Gaussian weighted least-squares fit is linear in
        ``q = 1 / sqrt(density)``.  Solving it directly is both more accurate
        and more robust than repeatedly optimizing density itself.
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        factors = []
        values = []
        weights = []
        for record in records:
            try:
                experiment = record['input']
                value = float(record['value'])
                sigma = float(record['sigma'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Malformed calibration record.') from exc

            if not np.isfinite(value) or not np.isfinite(sigma) or sigma <= 0:
                raise ValueError('Calibration values and uncertainties must be finite; sigma must be positive.')

            # ``mode`` is dimensionless, so this is the speed for q=1.
            factor = np.sqrt(MU * mode(experiment))
            factors.append(factor)
            values.append(value)
            weights.append(1.0 / sigma**2)

        factors = np.asarray(factors, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)

        numerator = np.sum(weights * factors * values)
        denominator = np.sum(weights * factors * factors)
        q = numerator / denominator

        # The stated prior range is a hard physical constraint.  Expressing
        # it in q keeps the fit a one-dimensional projection onto the allowed
        # interval while retaining the closed-form solution above.
        q_min = 1.0 / np.sqrt(DENSITY_MAX)
        q_max = 1.0 / np.sqrt(DENSITY_MIN)
        q = np.clip(q, q_min, q_max)
        self.density = float(1.0 / q**2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.density)
