"""Model for expansion of three hard-core bosons on an eight-site chain.

The Hamiltonian can be mapped to free fermions in one dimension, which is
particularly convenient for fitting the density data. Momentum, however,
depends on the bosonic one-body density matrix; its off-diagonal elements
contain the Jordan--Wigner string and are not the free-fermion correlator.
This module therefore evolves the state directly in the (small) hard-core
boson Hilbert space.
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


# The occupation-number basis is represented by an integer bit mask. The
# ordering is fixed so cached time evolutions are reproducible.
BASIS = tuple(sum(1 << site for site in sites) for sites in combinations(range(SITES), PARTICLES))
BASIS_INDEX = {mask: index for index, mask in enumerate(BASIS)}
INITIAL_INDEX = BASIS_INDEX[sum(1 << site for site in OCCUPIED)]


def hamiltonian(hopping, trap, tilt):
    """Return the single-particle Hamiltonian used by the apparatus."""

    hopping = float(hopping)
    trap = float(trap)
    tilt = float(tilt)
    matrix = np.diag(trap * POSITIONS**2 + tilt * POSITIONS).astype(float)
    matrix += np.diag(np.full(SITES - 1, -hopping), 1)
    matrix += np.diag(np.full(SITES - 1, -hopping), -1)
    return matrix


def many_body_hamiltonian(hopping, trap, tilt):
    """Construct the hard-core boson Hamiltonian in the three-particle basis."""

    hopping = float(hopping)
    trap = float(trap)
    tilt = float(tilt)
    matrix = np.zeros((len(BASIS), len(BASIS)), dtype=float)
    site_energy = trap * POSITIONS**2 + tilt * POSITIONS

    for column, mask in enumerate(BASIS):
        matrix[column, column] = sum(site_energy[site] for site in range(SITES) if mask & (1 << site))
        for site in range(SITES - 1):
            left = 1 << site
            right = 1 << (site + 1)
            # A hop is possible exactly when one end is occupied and the
            # other is empty. Hard-core boson hopping has unit matrix
            # element, so no occupation-number square-root is present.
            if bool(mask & left) != bool(mask & right):
                target = mask ^ left ^ right
                row = BASIS_INDEX[target]
                matrix[row, column] = -hopping
    return matrix


@lru_cache(maxsize=8192)
def eigensystem(hopping, trap, tilt):
    """Cached many-body eigensystem for a set of controls."""

    return eigh(many_body_hamiltonian(float(hopping), float(trap), float(tilt)))


@lru_cache(maxsize=8192)
def state(hopping, trap, tilt, duration):
    """Return the evolved state vector for the fixed initial occupation."""

    energies, vectors = eigensystem(float(hopping), float(trap), float(tilt))
    initial = vectors[INITIAL_INDEX, :].conj()
    return vectors @ (np.exp(-1j * energies * float(duration)) * initial)


@lru_cache(maxsize=8192)
def correlation_matrix(hopping, trap, tilt, duration):
    """Return ``<b_i^dagger b_j>`` for the evolved hard-core boson state."""

    wavefunction = state(float(hopping), float(trap), float(tilt), float(duration))
    matrix = np.zeros((SITES, SITES), dtype=complex)

    for mask, amplitude in zip(BASIS, wavefunction):
        occupied = [site for site in range(SITES) if mask & (1 << site)]
        for site in occupied:
            matrix[site, site] += amplitude.conj() * amplitude
        for source in occupied:
            for target in range(SITES):
                if mask & (1 << target):
                    continue
                target_mask = mask ^ (1 << source) ^ (1 << target)
                target_amplitude = wavefunction[BASIS_INDEX[target_mask]]
                matrix[target, source] += target_amplitude.conj() * amplitude

    return matrix


def predict_at(experiments, hopping):
    """Evaluate the documented density or momentum observable."""

    result = []
    for experiment in experiments:
        trap = float(experiment["trap"])
        tilt = float(experiment["tilt"])
        duration = float(experiment["duration"])
        matrix = correlation_matrix(float(hopping), trap, tilt, duration)

        if experiment["observable"] == "density":
            site = int(experiment["site"])
            value = matrix[site, site].real
        elif experiment["observable"] == "momentum":
            wave_number = float(experiment["wave_number"])
            mode = np.exp(-1j * wave_number * POSITIONS)
            # The 1/SITES factor gives the stipulated per-site normalization.
            value = np.vdot(mode, matrix @ mode).real / SITES
        else:
            raise ValueError("observable must be 'density' or 'momentum'")
        result.append(float(value))
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.hopping = None

    def fit(self, records):
        """Fit the common hopping from calibration records and return self."""

        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        observed = np.asarray([float(record["value"]) for record in records], dtype=float)
        sigma = np.asarray([float(record.get("sigma", 0.001)) for record in records], dtype=float)
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("calibration values and uncertainties must be finite, positive numbers")

        def objective(hopping):
            residual = (predict_at(experiments, hopping) - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.8, 1.2),
            method="bounded",
            options={"xatol": 1e-12},
        )
        # Explicit clipping keeps the documented hopping range true under
        # roundoff in the bounded optimizer.
        self.hopping = float(np.clip(result.x, 0.8, 1.2))
        return self

    def predict(self, experiments):
        if self.hopping is None:
            raise RuntimeError("Model.fit must be called before predict")
        return predict_at(experiments, self.hopping)
