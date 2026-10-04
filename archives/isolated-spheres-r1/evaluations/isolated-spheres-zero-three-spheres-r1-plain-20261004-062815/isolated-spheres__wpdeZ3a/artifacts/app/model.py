import numpy as np
from scipy.special import gammaln


def pair_force(separation, radius, cutoff=36):
    """Return the dimensionless classical force between two spheres.

    The zero-frequency scattering expression is a sum over azimuthal
    sectors.  ``coupling`` is the translation matrix between the regular
    multipoles of the two spheres.  Since both spheres are individually
    neutral, their monopole response is absent; consequently the multipole
    sum starts at degree one, including in the ``m == 0`` sector.
    """
    separation = float(separation)
    radius = float(radius)
    cutoff = int(cutoff)
    if separation <= 0.0 or radius < 0.0 or cutoff < 0:
        raise ValueError("invalid separation, radius, or multipole cutoff")

    ratio = radius / separation
    if ratio >= 1.0:
        raise ValueError("the spheres must not overlap")
    if ratio == 0.0:
        return 0.0

    result = 0.0
    log_ratio = np.log(ratio)
    for m in range(cutoff + 1):
        # Degree zero is the net-charge mode.  It is not thermally available
        # for either isolated, exactly neutral sphere.
        ell = np.arange(max(1, m), cutoff + 1, dtype=float)
        if ell.size == 0:
            continue
        left, right = ell[:, None], ell[None, :]
        degree = left + right + 1.0
        normalization = gammaln(left + right + 1.0) - 0.5 * (
            gammaln(left + m + 1.0) + gammaln(left - m + 1.0)
            + gammaln(right + m + 1.0) + gammaln(right - m + 1.0))
        coupling = np.exp(normalization + degree * log_ratio)
        derivative = -degree * coupling / separation
        matrix = np.eye(len(ell)) - coupling @ coupling
        result += (1.0 if m == 0 else 2.0) * np.trace(
            np.linalg.solve(matrix, coupling @ derivative))
    return float(result)


def predict_at(experiments, radius):
    result = []
    for e in experiments:
        if e['kind'] == 'dipole':
            result.append(radius**3 * e['field'])
        elif e['kind'] == 'force':
            result.append(pair_force(e['separation'], radius))
        else:
            raise ValueError(f"unknown experiment kind: {e['kind']!r}")
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.radius = None

    def fit(self, records):
        records = list(records)
        fields = np.asarray([r['input']['field'] for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if len(records) == 0 or np.any(sigma <= 0.0):
            raise ValueError("records must contain positive-uncertainty readings")

        # For an isolated conducting sphere, p/(4*pi*epsilon_0) = R^3 E.
        # The zero-field-subtracted response therefore passes through the
        # origin, and the known instrument uncertainties determine the WLS
        # estimate of its slope.
        weights = 1.0 / sigma**2
        denominator = np.sum(weights * fields**2)
        if denominator == 0.0:
            raise ValueError("at least one calibration field must be nonzero")
        slope = np.sum(weights * fields * values) / denominator
        if not np.isfinite(slope) or slope <= 0.0:
            raise ValueError("calibration does not imply a positive radius")
        self.radius = float(np.cbrt(slope))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.radius)
