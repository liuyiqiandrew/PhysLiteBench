import numpy as np
from scipy.special import gammaln


def pair_force(separation, radius, cutoff=36):
    """Return the classical force for two isolated, neutral spheres.

    The zero-frequency thermal interaction can be written as a multipole
    scattering determinant.  For two identical spheres, each azimuthal
    sector has a symmetric coupling matrix ``C`` and contributes

        trace((I - C C)^-1 C dC/dR)

    to the force (in units of ``k_B*T/ell``).  The ``l=0`` multipole is the
    net-charge degree of freedom.  It is absent here because both spheres
    are individually constrained to have zero charge; retaining it would
    instead describe electrically grounded spheres.

    ``cutoff`` is deliberately retained as part of the helper's API.  The
    documented geometry makes 36 multipoles more than sufficient while
    keeping the calculation inexpensive.
    """
    separation = float(separation)
    radius = float(radius)
    cutoff = int(cutoff)

    if separation <= 0.0 or radius <= 0.0:
        raise ValueError("separation and radius must be positive")
    if cutoff < 1:
        raise ValueError("cutoff must be at least one")

    result = 0.0
    log_ratio = np.log(radius / separation)

    for m in range(cutoff + 1):
        # m=0,l=0 is the monopole (total-charge) channel.  It must not be
        # included for spheres that are isolated and constrained to Q=0.
        first_degree = max(m, 1)
        ell = np.arange(first_degree, cutoff + 1, dtype=float)
        left, right = ell[:, None], ell[None, :]
        degree = left + right + 1.0

        # Translation coefficients in a normalized spherical-multipole
        # basis.  Using gammaln avoids factorial overflow.
        normalization = gammaln(degree) - 0.5 * (
            gammaln(left + m + 1.0) + gammaln(left - m + 1.0)
            + gammaln(right + m + 1.0) + gammaln(right - m + 1.0))
        coupling = np.exp(normalization + degree * log_ratio)
        derivative = -degree * coupling / separation

        system = np.eye(len(ell)) - coupling @ coupling
        result += (1.0 if m == 0 else 2.0) * np.trace(
            np.linalg.solve(system, coupling @ derivative)
        )

    return float(result)


def predict_at(experiments, radius):
    result = []
    for e in experiments:
        if e['kind'] == 'dipole':
            result.append(float(radius) ** 3 * float(e['field']))
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

        # For an isolated neutral conducting sphere, p/(4*pi*eps0) = a^3 E.
        # The zero-field mean has already been subtracted, so this is a
        # weighted least-squares fit through the origin.
        weights = 1.0 / sigma**2
        slope = np.sum(weights * fields * values) / np.sum(weights * fields**2)
        if not np.isfinite(slope) or slope <= 0.0:
            raise ValueError("calibration does not determine a positive radius")
        self.radius = float(np.cbrt(slope))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.radius)
