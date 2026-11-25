"""Model for the elastic bulk-wave calibration problem.

The constitutive energy is a scalar multiple of a dimensionless energy, so
the acoustic tensor is independent of ``modulus``.  Consequently every phase
speed has the form ``sqrt(modulus) * speed_at_unit_modulus``; this makes the
one-parameter calibration a weighted linear least-squares problem in
``sqrt(modulus)``.
"""

import numpy as np


RHO0 = 1000.0
_LOWER_MODULUS = 0.8
_UPPER_MODULUS = 1.6


def geometry(experiment):
    """Return the deformation gradient and a unit current-space direction."""

    deformation = np.asarray(experiment["deformation"], dtype=float)
    direction = np.asarray(experiment["direction"], dtype=float)
    norm = np.linalg.norm(direction)
    if deformation.shape != (3, 3):
        raise ValueError("deformation must have shape (3, 3)")
    if direction.shape != (3,) or not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must be a nonzero three-vector")
    return deformation, direction / norm


def operator(experiment):
    """Build the unit-modulus spatial acoustic tensor and current density.

    Let ``C = F.T @ F`` and ``G = C^{-1}``.  For the dimensionless energy in
    the README, the tangent with respect to ``C`` is

        0.8 (I⊗I - sym(I⊗I))
        + 2 G⊗G - (2 log(J) - 1.4) sym(G⊗G).

    The factor of four from the two derivatives of ``C = F.T F`` is included
    in these coefficients.  Pushing this tangent forward along the current
    propagation direction gives the acoustic tensor.  The density is the
    current density, ``rho0 / J``, because mass is conserved.
    """

    deformation, direction = geometry(experiment)
    volume = float(np.linalg.det(deformation))
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("deformation must have positive determinant")

    metric = deformation.T @ deformation
    metric_inverse = np.linalg.inv(metric)
    identity = np.eye(3)

    # T[a,b,c,d] is the constitutive tangent in metric (C) indices.  Keeping
    # the tensor explicit makes the index order in the push-forward clear.
    tensor = 0.8 * (
        np.einsum("ab,cd->abcd", identity, identity)
        - 0.5
        * (
            np.einsum("ac,bd->abcd", identity, identity)
            + np.einsum("ad,bc->abcd", identity, identity)
        )
    )
    coefficient = 2.0 * np.log(volume) - 1.4
    tensor += 2.0 * np.einsum(
        "ab,cd->abcd", metric_inverse, metric_inverse
    )
    tensor -= coefficient * (
        np.einsum("ac,bd->abcd", metric_inverse, metric_inverse)
        + np.einsum("ad,bc->abcd", metric_inverse, metric_inverse)
    )

    material_direction = deformation.T @ direction
    acoustic = np.einsum(
        "ib,a,bacd,c,jd->ij",
        deformation,
        material_direction,
        tensor,
        material_direction,
        deformation,
        optimize=True,
    ) / volume
    acoustic = 0.5 * (acoustic + acoustic.T)
    current_density = RHO0 / volume
    return acoustic, current_density


def predict_at(experiments, modulus):
    """Predict phase speeds for ``experiments`` at a specified modulus."""

    modulus = float(modulus)
    if not np.isfinite(modulus) or modulus <= 0.0:
        raise ValueError("modulus must be a positive finite number")

    result = []
    for experiment in experiments:
        acoustic, density = operator(experiment)
        eigenvalues = np.linalg.eigvalsh(acoustic)
        branch = int(experiment["branch"])
        if branch not in (0, 1, 2):
            raise ValueError("branch must be 0, 1, or 2")
        eigenvalue = eigenvalues[branch]
        # The stated apparatus has propagating bulk modes.  The small clamp
        # only protects against round-off at a zero eigenvalue.
        if eigenvalue < 0.0:
            raise ValueError("acoustic tensor has a negative mode")
        result.append(np.sqrt(max(0.0, 1e6 * modulus * eigenvalue / density)))
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.modulus = None

    def fit(self, records):
        """Fit the shared modulus from calibration records and return self."""

        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        unit_speeds = predict_at(
            (record["input"] for record in records), 1.0
        )
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if (
            not np.all(np.isfinite(values))
            or not np.all(np.isfinite(sigma))
            or np.any(sigma <= 0.0)
        ):
            raise ValueError("record values and sigmas must be finite; sigmas positive")

        weights = 1.0 / sigma
        design = unit_speeds * weights
        observations = values * weights
        scale = np.dot(design, observations) / np.dot(design, design)
        scale = np.clip(
            scale, np.sqrt(_LOWER_MODULUS), np.sqrt(_UPPER_MODULUS)
        )
        self.modulus = float(scale * scale)
        return self

    def predict(self, experiments):
        """Return phase speeds as a one-dimensional NumPy array."""

        if self.modulus is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.modulus)
