"""Large-deviation model for the heat dissipated by the circulating trap."""

import numpy as np


def _heat_rate_per_relaxation_time(drive_ratio, relative_heat):
    """Return the heat rate function in units of ``mobility * stiffness``.

    With ``r = drive_ratio`` and relaxation time ``tau = mobility *
    stiffness * t``, the mean heat is ``2*r**2*tau``.  The tilted heat
    generator has bulk eigenvalue

        1 - sqrt(1 + 4*r**2*s*(1-s)).

    Heat is work minus the trap-energy change.  Integrability of the two
    endpoint factors restricts that eigenvalue's counting-field interval.
    Its Legendre transform is therefore the work rate in the middle, with a
    negative-heat linear tail, and (for weak enough circulation) a positive
    linear tail as well.
    """
    r = float(drive_ratio)
    z = float(relative_heat)
    u = r * r
    q = 2.0 * u * z

    # Interior Legendre transform of the tilted-generator eigenvalue.
    work_rate = np.sqrt((1.0 + u) * (q * q + 4.0 * u) / (4.0 * u)) - q / 2.0 - 1.0

    # The final-energy endpoint makes the admissible field end at s=1.
    # Its supporting line takes over for q <= -2*u (equivalently z <= -1).
    if q <= -2.0 * u:
        return -q

    # For u < 1/3 the initial-energy endpoint also truncates the negative
    # counting-field side.  The crossover is q_c below; using the supporting
    # line avoids extrapolating the work formula into inadmissible fields.
    if u < 1.0 / 3.0:
        q_c = 2.0 * u * (3.0 - u) / (1.0 - 3.0 * u)
        if q >= q_c:
            return ((1.0 - u) * q - 4.0 * u) / (1.0 + u)

    return work_rate


def response(e):
    """Heat rate for unit mobility for one public experiment dictionary."""
    return float(e['stiffness']) * _heat_rate_per_relaxation_time(
        e['drive_ratio'], e['relative_heat']
    )


def predict_at(experiments, mobility):
    """Evaluate the public response at a supplied common mobility."""
    return float(mobility) * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by inverse-variance weighted least squares."""
        if not records:
            raise ValueError('at least one calibration record is required')

        unit_response = np.asarray([response(record['input']) for record in records])
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(unit_response).all() or not np.isfinite(observed).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0.0)):
            raise ValueError('calibration values and standard deviations must be finite, with sigma > 0')

        weights = 1.0 / (sigma * sigma)
        denominator = np.sum(weights * unit_response * unit_response)
        if denominator <= 0.0:
            raise ValueError('calibration records do not identify mobility')
        estimate = np.sum(weights * unit_response * observed) / denominator

        # The apparatus specifies this as the physically admissible interval.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.mobility)
