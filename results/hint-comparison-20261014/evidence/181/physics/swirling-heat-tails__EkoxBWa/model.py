import numpy as np


def response(e):
    """Rate function per unit mobility for the reported heat.

    The tilted generator for the heat has (writing ``s`` for the counting
    field) the principal eigenvalue

        mu * stiffness * (1 - sqrt(1 - 4*r**2*s*(1+s))).

    The square-root branch by itself is the rate function for the central
    part of the distribution.  Heat also contains the change in trap
    energy, however.  The stationary initial density and the unconstrained
    final position multiply the tilted eigenfunctions by two endpoint
    Gaussian factors.  Their integrability restricts the counting field to
    s > -1 and, when r < 1/sqrt(3), to

        s < (1-r**2)/(1+r**2).

    The corresponding Legendre transform therefore has linear endpoint
    tails.  Keeping those tails is important on the full requested z
    interval.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    r2 = r * r

    # The unconstrained Legendre transform of the SCGF.  This is written in
    # a form that is well behaved for both small and large drive ratios.
    central = np.sqrt((1.0 + r2) * (1.0 + r2 * z * z)) - r2 * z - 1.0

    # A final energy of order t gives the negative heat tail.  Its endpoint
    # counting-field boundary is s=-1, whose tangent has slope z=-1.
    if z <= -1.0:
        rate = -2.0 * r2 * z
    else:
        rate = central

        # For weak circulation the initial-energy endpoint is reached before
        # the square-root branch point.  Its tangent supplies the positive
        # heat tail.  At and above 1/sqrt(3) the branch point is reached only
        # at infinite z, so there is no finite-z upper linear tail.
        if r2 < 1.0 / 3.0:
            s_edge = (1.0 - r2) / (1.0 + r2)
            delta_edge = (1.0 - 3.0 * r2) / (1.0 + r2)
            z_edge = (3.0 - r2) / (1.0 - 3.0 * r2)
            if z >= z_edge:
                rate = 2.0 * r2 * s_edge * z - 1.0 + delta_edge

    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # All preparations share one mobility and the supplied uncertainties
        # are independent and known, so weighted least squares is the Gaussian
        # maximum-likelihood estimate for the linear response.
        x = np.asarray([response(record['input']) for record in records], dtype=float)
        y = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        estimate = np.sum((x * y) / (sigma * sigma)) / np.sum(
            (x * x) / (sigma * sigma)
        )
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
