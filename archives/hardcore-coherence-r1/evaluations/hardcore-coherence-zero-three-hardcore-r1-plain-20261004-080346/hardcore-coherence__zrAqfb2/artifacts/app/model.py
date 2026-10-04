import itertools
from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


SITES = 8
PARTICLES = 3
OCCUPIED = (2, 3, 4)
POSITIONS = np.arange(SITES, dtype=float) - 3.5


def hamiltonian(hopping, trap, tilt):
    """Return the one-particle Hamiltonian on the open chain."""
    matrix = np.diag(trap * POSITIONS**2 + tilt * POSITIONS).astype(float)
    matrix += np.diag(np.full(SITES - 1, -hopping, dtype=float), 1)
    matrix += np.diag(np.full(SITES - 1, -hopping, dtype=float), -1)
    return matrix


# A hard-core boson state is represented by the bit mask of its occupied
# sites.  There are only C(8, 3) = 56 states, so exact many-body evolution is
# inexpensive and gives the correct bosonic momentum correlations.
_BASIS = tuple(
    sum(1 << site for site in sites)
    for sites in itertools.combinations(range(SITES), PARTICLES)
)
_STATE_INDEX = {mask: index for index, mask in enumerate(_BASIS)}
_INITIAL_MASK = sum(1 << site for site in OCCUPIED)
_INITIAL_INDEX = _STATE_INDEX[_INITIAL_MASK]


@lru_cache(maxsize=8192)
def _many_body_eigensystem(hopping, trap, tilt):
    """Diagonalize the fixed-three-particle hard-core-boson Hamiltonian."""
    matrix = np.zeros((len(_BASIS), len(_BASIS)), dtype=float)

    for column, mask in enumerate(_BASIS):
        matrix[column, column] = sum(
            trap * POSITIONS[site] ** 2 + tilt * POSITIONS[site]
            for site in range(SITES)
            if mask & (1 << site)
        )
        for site in range(SITES - 1):
            left_occupied = bool(mask & (1 << site))
            right_occupied = bool(mask & (1 << (site + 1)))
            if left_occupied != right_occupied:
                target = mask ^ (1 << site) ^ (1 << (site + 1))
                matrix[_STATE_INDEX[target], column] = -hopping

    return eigh(matrix)


@lru_cache(maxsize=8192)
def _state(hopping, trap, tilt, duration):
    energies, vectors = _many_body_eigensystem(hopping, trap, tilt)
    coefficients = vectors[_INITIAL_INDEX, :].conj()
    return vectors @ (coefficients * np.exp(-1j * energies * duration))


@lru_cache(maxsize=8192)
def correlation_matrix(hopping, trap, tilt, duration):
    """Return rho[i, j] = <b_i^dagger b_j> at the requested time."""
    state = _state(float(hopping), float(trap), float(tilt), float(duration))
    matrix = np.zeros((SITES, SITES), dtype=complex)

    for i in range(SITES):
        for j in range(SITES):
            if i == j:
                matrix[i, i] = sum(
                    abs(amplitude) ** 2
                    for amplitude, mask in zip(state, _BASIS)
                    if mask & (1 << i)
                )
                continue

            i_bit = 1 << i
            j_bit = 1 << j
            matrix[i, j] = sum(
                state[_STATE_INDEX[mask ^ i_bit ^ j_bit]].conj() * amplitude
                for amplitude, mask in zip(state, _BASIS)
                if mask & j_bit and not (mask & i_bit)
            )

    # Remove only numerical anti-Hermitian roundoff.
    return (matrix + matrix.conj().T) / 2.0


def predict_at(experiments, hopping):
    """Evaluate density or released momentum observables in input order."""
    result = []
    hopping = float(hopping)

    for experiment in experiments:
        matrix = correlation_matrix(
            hopping,
            float(experiment["trap"]),
            float(experiment["tilt"]),
            float(experiment["duration"]),
        )
        if experiment["observable"] == "density":
            result.append(float(matrix[int(experiment["site"]), int(experiment["site"])].real))
        elif experiment["observable"] == "momentum":
            mode = np.exp(-1j * float(experiment["wave_number"]) * POSITIONS)
            # The factor 1/SITES gives the stated average PARTICLES/SITES.
            result.append(float(np.vdot(mode, matrix @ mode).real / SITES))
        else:
            raise ValueError(f"unknown observable: {experiment['observable']!r}")

    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.hopping = None

    def fit(self, records):
        """Fit the common hopping using the supplied Gaussian uncertainties."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([float(record["value"]) for record in records])
        sigmas = np.asarray([float(record.get("sigma", 0.001)) for record in records])
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
            raise ValueError("calibration values and sigmas must be finite, with positive sigma")

        def objective(hopping):
            residual = (predict_at(experiments, hopping) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.8, 1.2),
            method="bounded",
            options={"xatol": 1e-11},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError("unable to fit hopping")
        self.hopping = float(np.clip(result.x, 0.8, 1.2))
        return self

    def predict(self, experiments):
        if self.hopping is None:
            raise RuntimeError("fit the model before prediction")
        return predict_at(experiments, self.hopping)
