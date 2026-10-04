import numpy as np
from scipy.special import gammaln


def pair_force(separation, radius, cutoff=36):
    """Return the dimensionless classical force between two neutral spheres.

    At zero frequency the electrostatic partition function can be written as a
    determinant over spherical multipoles.  Since each sphere has fixed zero
    total charge, its monopole is not a fluctuating degree of freedom.  Thus
    the determinant starts at ``ell=1``; retaining ``ell=0`` would instead
    describe spheres whose charges can fluctuate through a reservoir.

    ``separation`` is the center-to-center distance and the returned value is
    in units of ``k_B*T/ell``.  The two signs of every nonzero azimuthal index
    are accounted for by the factor of two below.
    """
    separation = float(separation)
    radius = float(radius)
    cutoff = int(cutoff)

    if separation <= 0.0 or radius < 0.0 or cutoff < 0:
        raise ValueError("invalid geometry or multipole cutoff")
    if radius >= separation:
        raise ValueError("spheres must not overlap")
    if radius == 0.0 or cutoff == 0:
        return 0.0

    result = 0.0
    log_radius_ratio = np.log(radius / separation) if radius else -np.inf

    for m in range(cutoff + 1):
        # ell=0 is the total-charge (monopole) mode.  It must be removed for
        # individually isolated, exactly neutral conductors.
        ell = np.arange(max(1, m), cutoff + 1, dtype=float)
        left, right = ell[:, None], ell[None, :]
        degree = left + right + 1.0
        normalization = gammaln(left + right + 1.0) - 0.5 * (
            gammaln(left + m + 1.0) + gammaln(left - m + 1.0)
            + gammaln(right + m + 1.0) + gammaln(right - m + 1.0))
        coupling = np.exp(normalization + degree * log_radius_ratio)
        derivative = -degree * coupling / separation
        matrix = np.eye(len(ell)) - coupling @ coupling
        result += (1 if m == 0 else 2) * np.trace(
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
    return np.asarray(result)


class Model:
    def __init__(self):
        self.radius = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        fields = np.array([r['input']['field'] for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        slope = np.sum(fields * values / sigma**2) / np.sum(fields**2 / sigma**2)
        self.radius = float(np.cbrt(slope))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.radius)
