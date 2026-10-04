"""Model for expansion of three hard-core bosons on an eight-site chain.

The Hamiltonian is quadratic after a Jordan--Wigner transformation, but the
momentum readout is a bosonic correlation function.  Consequently, using the
single-particle fermionic correlation matrix directly for momentum would give
the wrong answer.  The Hilbert space here is small (C(8, 3) = 56), so this
module evolves the exact hard-core-boson state and evaluates the readouts
directly in that basis.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


SITES = 8
PARTICLES = 3
OCCUPIED = (2, 3, 4)
POSITIONS = np.arange(SITES, dtype=float) - 3.5


# A fixed basis makes the many-body Hamiltonian construction deterministic.
_BASIS = tuple(
    sum(1 << site for site in occupied)
    for occupied in combinations(range(SITES), PARTICLES)
)
_BASIS_INDEX = {state: index for index, state in enumerate(_BASIS)}
_INITIAL_STATE = sum(1 << site for site in OCCUPIED)


def hamiltonian(hopping, trap, tilt):
    """Return the one-particle Hamiltonian retained for API compatibility."""
    matrix = np.diag(trap * POSITIONS**2 + tilt * POSITIONS).astype(float)
    matrix += np.diag(np.full(SITES - 1, -hopping), 1)
    matrix += np.diag(np.full(SITES - 1, -hopping), -1)
    return matrix


def _many_body_hamiltonian(hopping, trap, tilt):
    """Construct the hard-core-boson Hamiltonian in the 3-particle basis."""
    dimension = len(_BASIS)
    matrix = np.zeros((dimension, dimension), dtype=float)
    potential = trap * POSITIONS**2 + tilt * POSITIONS

    for column, state in enumerate(_BASIS):
        occupied_sites = [site for site in range(SITES) if state & (1 << site)]
        matrix[column, column] = sum(potential[site] for site in occupied_sites)

        # Nearest-neighbour hard-core boson hopping has unit matrix element.
        for site in range(SITES - 1):
            left = 1 << site
            right = 1 << (site + 1)
            if state & left and not state & right:
                target = state ^ left ^ right
                row = _BASIS_INDEX[target]
                matrix[row, column] = -hopping
            elif state & right and not state & left:
                target = state ^ left ^ right
                row = _BASIS_INDEX[target]
                matrix[row, column] = -hopping

    return matrix


@lru_cache(maxsize=8192)
def correlation_matrix(hopping, trap, tilt, duration):
    """Return C[i, j] = <b_i^dagger b_j> after the specified evolution."""
    many_body = _many_body_hamiltonian(hopping, trap, tilt)
    energies, vectors = eigh(many_body)

    initial_index = _BASIS_INDEX[_INITIAL_STATE]
    initial_in_eigenbasis = vectors[initial_index, :]
    state = vectors @ (np.exp(-1j * energies * duration) * initial_in_eigenbasis)

    correlation = np.zeros((SITES, SITES), dtype=complex)
    for i in range(SITES):
        bit_i = 1 << i
        for j in range(SITES):
            bit_j = 1 << j
            if i == j:
                correlation[i, j] = sum(
                    abs(state[index]) ** 2
                    for index, occupation in enumerate(_BASIS)
                    if occupation & bit_i
                )
                continue

            # b_i^dagger b_j maps a state with j occupied and i empty to the
            # state obtained by moving that particle from j to i.
            total = 0j
            for index, occupation in enumerate(_BASIS):
                if occupation & bit_j and not occupation & bit_i:
                    target = occupation ^ bit_j ^ bit_i
                    target_index = _BASIS_INDEX[target]
                    total += state[target_index].conjugate() * state[index]
            correlation[i, j] = total

    # Roundoff can leave a tiny anti-Hermitian component in the result.  The
    # physical correlation matrix is Hermitian, so remove that numerical noise.
    return (correlation + correlation.conj().T) / 2


def predict_at(experiments, hopping):
    """Evaluate an iterable of documented experiment dictionaries."""
    result = []
    for experiment in experiments:
        trap = float(experiment["trap"])
        tilt = float(experiment["tilt"])
        duration = float(experiment["duration"])
        matrix = correlation_matrix(float(hopping), trap, tilt, duration)
        observable = experiment["observable"]

        if observable == "density":
            site = int(experiment["site"])
            if not 0 <= site < SITES:
                raise ValueError("density site must be an integer from 0 through 7")
            value = matrix[site, site].real
        elif observable == "momentum":
            wave_number = float(experiment["wave_number"])
            mode = np.exp(-1j * wave_number * POSITIONS)
            # The factor 1/SITES gives the requested per-site normalization.
            value = np.vdot(mode, matrix @ mode).real / SITES
        else:
            raise ValueError("observable must be 'density' or 'momentum'")

        result.append(float(value))

    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.hopping=None

    def fit(self,records):
        """Fit the common hopping by weighted least squares and return self."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record.get("sigma", 0.001) for record in records], dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(sigmas)):
            raise ValueError("calibration values and uncertainties must be finite")
        if np.any(sigmas <= 0):
            raise ValueError("calibration uncertainties must be positive")

        def objective(hopping):
            prediction = predict_at(experiments, float(hopping))
            residual = (prediction - values) / sigmas
            return float(np.mean(residual**2))

        result = minimize_scalar(
            objective,
            bounds=(0.8, 1.2),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError("unable to fit the hopping parameter")

        self.hopping = float(np.clip(result.x, 0.8, 1.2))
        return self

    def predict(self,experiments):
        if self.hopping is None:
            raise RuntimeError('fit the model before calling predict')
        return predict_at(experiments,self.hopping)
