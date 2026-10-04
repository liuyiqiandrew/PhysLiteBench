"""Exact density dynamics and calibration for the eight-site ring.

The particles are hard-core bosons, so the evolution is performed in the
fixed-particle-number occupation basis rather than by evolving independent
single-particle states.  This distinction matters on a ring: hard-core
bosons have ordinary bosonic exchange signs, including across the boundary.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


SITES = 8
OCCUPIED = {2: (3, 4), 3: (2, 3, 4), 4: (2, 3, 4, 5)}


def potential(amplitude, asymmetry):
    """Return the on-site potential for the supplied controls."""

    angle = 2 * np.pi * np.arange(SITES) / SITES
    return amplitude * np.cos(angle) + asymmetry * np.sin(angle)


@lru_cache(maxsize=None)
def _basis(number):
    """Return occupation masks and their index lookup for ``number`` bosons."""

    states = tuple(
        sum(1 << site for site in occupied)
        for occupied in combinations(range(SITES), number)
    )
    return states, {mask: index for index, mask in enumerate(states)}


@lru_cache(maxsize=16384)
def _hamiltonian(hopping, number, flux, amplitude, asymmetry):
    """Construct the exact hard-core-boson Hamiltonian in the Fock basis."""

    states, index_of = _basis(number)
    site_potential = potential(amplitude, asymmetry)
    dimension = len(states)
    matrix = np.zeros((dimension, dimension), dtype=complex)

    for column, mask in enumerate(states):
        matrix[column, column] = sum(
            site_potential[site] for site in range(SITES) if mask & (1 << site)
        )

        for site in range(SITES):
            next_site = (site + 1) % SITES

            # b_site^dagger b_next_site moves a boson from next_site to
            # site. Hard-core bosons have no fermionic exchange sign.
            if mask & (1 << next_site) and not mask & (1 << site):
                moved = (mask | (1 << site)) & ~(1 << next_site)
                row = index_of[moved]
                forward = -hopping * np.exp(1j * flux / SITES)
                matrix[row, column] += forward
                matrix[column, row] += forward.conjugate()

    return matrix


@lru_cache(maxsize=16384)
def density(hopping, number, flux, amplitude, asymmetry, duration):
    """Return the occupation of every site after one prepared evolution."""

    number = int(number)
    states, index_of = _basis(number)
    energies, vectors = eigh(
        _hamiltonian(
            float(hopping),
            number,
            float(flux),
            float(amplitude),
            float(asymmetry),
        )
    )

    initial_mask = sum(1 << site for site in OCCUPIED[number])
    initial = np.zeros(len(states), dtype=complex)
    initial[index_of[initial_mask]] = 1.0
    coefficients = vectors.conj().T @ initial
    evolved = vectors @ (coefficients * np.exp(-1j * energies * float(duration)))

    probabilities = np.abs(evolved) ** 2
    return np.asarray(
        [
            sum(
                probabilities[index]
                for index, mask in enumerate(states)
                if mask & (1 << site)
            )
            for site in range(SITES)
        ],
        dtype=float,
    )


def predict_at(experiments, hopping):
    """Predict site occupations in the same order as ``experiments``."""

    if hopping is None:
        raise RuntimeError("Model must be fit before predict is called.")

    predictions = []
    for experiment in experiments:
        number = int(experiment["number"])
        flux = float(experiment["flux"])
        amplitude = float(experiment["amplitude"])
        asymmetry = float(experiment["asymmetry"])
        duration = float(experiment["duration"])
        site = int(experiment["site"])
        predictions.append(
            density(
                float(hopping),
                number,
                flux,
                amplitude,
                asymmetry,
                duration,
            )[site]
        )
    return np.asarray(predictions, dtype=float)


class Model:
    def __init__(self):
        self.hopping = None

    def fit(self, records):
        """Fit the common hopping from independent Gaussian calibration data."""

        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")

        experiments = [record["input"] for record in records]
        values = np.asarray([float(record["value"]) for record in records])
        sigmas = np.asarray([float(record["sigma"]) for record in records])
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
            raise ValueError("Calibration values and uncertainties must be finite, positive data.")

        def objective(hopping):
            residual = (predict_at(experiments, hopping) - values) / sigmas
            return float(np.mean(residual * residual))

        result = minimize_scalar(
            objective,
            bounds=(0.8, 1.2),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError("Unable to fit the hopping parameter.")

        self.hopping = float(np.clip(result.x, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.hopping)
