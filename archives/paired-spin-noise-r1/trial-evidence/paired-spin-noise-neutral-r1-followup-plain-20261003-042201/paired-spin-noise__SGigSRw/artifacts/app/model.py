"""Model for the three-site local-spin detector."""

import numpy as np
from scipy.linalg import eigh


_N_MODES = 6
_DIMENSION = 1 << _N_MODES
_PAIRING_FACTORS = np.array([1.0, 0.85, 1.15])
_PAIRING_PHASES = np.array([0.0, 1.0, -0.4])


def _annihilation(mode):
    """Return the fermionic annihilation matrix for one mode."""
    operator = np.zeros((_DIMENSION, _DIMENSION), dtype=complex)
    lower_bits = (1 << mode) - 1
    for state in range(_DIMENSION):
        if state & (1 << mode):
            sign = -1.0 if (state & lower_bits).bit_count() % 2 else 1.0
            operator[state ^ (1 << mode), state] = sign
    return operator


_ANNIHILATORS = tuple(_annihilation(mode) for mode in range(_N_MODES))
_CREATORS = tuple(operator.conj().T for operator in _ANNIHILATORS)
_NUMBER_OPERATORS = tuple(
    creator @ annihilator
    for creator, annihilator in zip(_CREATORS, _ANNIHILATORS)
)


def _single_particle_matrix(experiment):
    hopping = float(experiment["hopping"])
    offset = float(experiment["offset"])
    h = np.diag(np.array([-0.35, 0.15, 0.70]) + offset)
    for i, j, scale in ((0, 1, 1.0), (1, 2, 0.8), (0, 2, 0.3)):
        h[i, j] = h[j, i] = -hopping * scale
    return h


def energy_matrix(experiment):
    """Construct the many-body Hamiltonian on the 64-state Fock space."""
    h = _single_particle_matrix(experiment)
    hamiltonian = np.zeros((_DIMENSION, _DIMENSION), dtype=complex)

    # The mode ordering is (site 0 up, site 0 down, site 1 up, ...).
    for spin in (0, 1):
        for i in range(3):
            for j in range(3):
                hamiltonian += h[i, j] * (
                    _CREATORS[2 * i + spin] @ _ANNIHILATORS[2 * j + spin]
                )

    pairing = float(experiment["pairing"])
    phase = float(experiment["phase"])
    for site, (factor, phase_factor) in enumerate(
        zip(_PAIRING_FACTORS, _PAIRING_PHASES)
    ):
        delta = pairing * factor * np.exp(1j * phase * phase_factor)
        create_pair = _CREATORS[2 * site] @ _CREATORS[2 * site + 1]
        hamiltonian += delta * create_pair + delta.conjugate() * create_pair.conj().T

    return (hamiltonian + hamiltonian.conj().T) * 0.5


def _spin_operator(site):
    return 0.5 * (
        _NUMBER_OPERATORS[2 * site] - _NUMBER_OPERATORS[2 * site + 1]
    )


def spectrum(experiment):
    """Return transition energies and thermal transition weights."""
    energies, states = eigh(energy_matrix(experiment), check_finite=True)
    shifted = energies - energies.min()
    probabilities = np.exp(-shifted / float(experiment["temperature"]))
    probabilities /= probabilities.sum()

    spin_in_energy_basis = (
        states.conj().T @ _spin_operator(int(experiment["site"])) @ states
    )
    weights = np.abs(spin_in_energy_basis) ** 2 * probabilities[None, :]
    frequencies = energies[:, None] - energies[None, :]
    return frequencies, weights


def response(experiment):
    """Calculate the detector response before applying its unknown scale."""
    frequency, weight = spectrum(experiment)
    center = float(experiment["center"])
    width = float(experiment["width"])
    detector_weight = np.zeros_like(frequency)
    positive = frequency > 0.0
    detector_weight[positive] = frequency[positive] ** 2 * np.exp(
        -0.5 * ((frequency[positive] - center) / width) ** 2
    )
    return float(np.sum(weight * detector_weight))


def predict_at(experiments, gain):
    return float(gain) * np.asarray(
        [response(experiment) for experiment in experiments], dtype=float
    )


class Model:
    def __init__(self):
        self.gain = None

    def fit(self, records):
        """Fit the common detector scale by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("fit requires at least one calibration record")

        responses = np.asarray([response(record["input"]) for record in records])
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if not np.all(np.isfinite(responses + values + sigmas)) or np.any(sigmas <= 0):
            raise ValueError("calibration values and uncertainties must be finite")

        weights = 1.0 / sigmas**2
        denominator = np.sum(weights * responses**2)
        if denominator <= 0.0:
            raise ValueError("calibration records do not identify a detector scale")
        fitted_gain = np.sum(weights * responses * values) / denominator
        self.gain = float(np.clip(fitted_gain, 0.6, 1.6))
        return self

    def predict(self, experiments):
        if self.gain is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.gain)
