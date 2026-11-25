import numpy as np
from scipy.optimize import brentq

MU = 1e6


def geometry(e):
    s = e['stretch']
    pressure = e['pressure'] / MU
    t = brentq(lambda t: t*t - 1 + 2*np.log(s*t) + pressure*s*t, .2, 2.)
    return s, t


def surface_matrix(x, s, t):
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
    s, t = geometry(e)
    def secular(x):
        matrix = surface_matrix(x, s, t)
        return np.linalg.det(matrix) / (t**4*x)
    x = brentq(secular, 1e-6, 1-1e-10, xtol=1e-13)
    return s*s - t*t + t*t*x


def predict_at(experiments, density):
    return np.sqrt(MU*np.array([mode(e) for e in experiments])/density)


class Model:
    def __init__(self):
        self.density = None

    def fit(self, records):
        """Infer the (reference) density from calibration measurements.

        For a fixed experiment the wave calculation gives a quantity ``a``
        such that the predicted speed is

            speed = a / sqrt(density).

        Thus ``q = 1/sqrt(density)`` is a linear parameter.  Fitting ``q``
        directly avoids a less well-conditioned nonlinear optimization and
        also makes use of the individual measurement uncertainties.
        """
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        inputs = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        if (not np.isfinite(values).all() or
                not np.isfinite(sigmas).all() or
                np.any(sigmas <= 0)):
            raise ValueError('calibration values and sigmas must be finite, with sigma > 0')

        # ``a`` has units m/s * sqrt(kg/m^3), and contains all of the
        # experiment-dependent deformation and boundary-condition physics.
        a = np.asarray([np.sqrt(MU * mode(experiment)) for experiment in inputs])
        if not np.isfinite(a).all() or np.any(a <= 0):
            raise ValueError('calibration inputs produced an invalid wave speed')

        weights = 1.0 / (sigmas * sigmas)
        q = np.sum(weights * a * values) / np.sum(weights * a * a)

        # The prior interval is part of the apparatus specification.  The
        # clipping also keeps predictions well-defined for an outlying or
        # unusually small calibration set.
        q_min = 1.0 / np.sqrt(1300.0)
        q_max = 1.0 / np.sqrt(900.0)
        q = float(np.clip(q, q_min, q_max))
        self.density = 1.0 / (q * q)
        return self

    def predict(self, experiments):
        if self.density is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(list(experiments), self.density)
