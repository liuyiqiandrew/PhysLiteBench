"""Model for the three-site local spin detector.

The sample has six fermionic modes, so its complete Fock space has only
``2**6 == 64`` states. Working in that space keeps the anomalous terms from
the pairing Hamiltonian in the spin correlation function.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh


_N_MODES = 6
_DIMENSION = 1 << _N_MODES
_PAIR_SCALES = np.array([1.0, 0.85, 1.15])
_PAIR_PHASES = np.array([0.0, 1.0, -0.4])


def _annihilation(mode):
    """Return the Jordan-Wigner annihilation matrix for one mode."""

    operator = np.zeros((_DIMENSION, _DIMENSION), dtype=complex)
    lower_states = (1 << mode) - 1
    for state in range(_DIMENSION):
        if state & (1 << mode):
            sign = -1 if (state & lower_states).bit_count() % 2 else 1
            operator[state ^ (1 << mode), state] = sign
    return operator


@lru_cache(maxsize=1)
def _fermion_operators():
    """Construct and cache the fixed operators used by every experiment."""

    annihilators = tuple(_annihilation(mode) for mode in range(_N_MODES))
    creators = tuple(operator.conj().T for operator in annihilators)

    bilinears = np.empty((_N_MODES, _N_MODES, _DIMENSION, _DIMENSION), complex)
    for i in range(_N_MODES):
        for j in range(_N_MODES):
            bilinears[i, j] = creators[i] @ annihilators[j]

    pair_creation = []
    pair_annihilation = []
    for site in range(3):
        up, down = 2 * site, 2 * site + 1
        pair_creation.append(creators[up] @ creators[down])
        pair_annihilation.append(annihilators[down] @ annihilators[up])

    spin = []
    for site in range(3):
        up, down = 2 * site, 2 * site + 1
        spin.append(0.5 * (bilinears[up, up] - bilinears[down, down]))

    return bilinears, tuple(pair_creation), tuple(pair_annihilation), tuple(spin)


def energy_matrix(e):
    """Return the 12-by-12 BdG matrix associated with an input.

    This helper is retained for compatibility with the original module. The
    prediction itself uses the complete Fock-space Hamiltonian below.
    """

    h_site = np.diag(np.array([-0.35, 0.15, 0.70]) + float(e["offset"]))
    for i, j, scale in ((0, 1, 1.0), (1, 2, 0.8), (0, 2, 0.3)):
        h_site[i, j] = h_site[j, i] = -float(e["hopping"]) * scale
    h = np.kron(h_site, np.eye(2))

    pair = np.zeros((_N_MODES, _N_MODES), dtype=complex)
    amplitudes = float(e["pairing"]) * _PAIR_SCALES * np.exp(
        1j * float(e["phase"]) * _PAIR_PHASES
    )
    for site, amplitude in enumerate(amplitudes):
        up, down = 2 * site, 2 * site + 1
        pair[up, down] = amplitude
        pair[down, up] = -amplitude
    return np.block([[h, pair], [-pair.conj(), -h.T]])


def _sample_hamiltonian(e):
    """Build the physical Hamiltonian on the 64-dimensional Fock space."""

    bilinears, pair_creation, pair_annihilation, _ = _fermion_operators()

    h_site = np.diag(np.array([-0.35, 0.15, 0.70]) + float(e["offset"]))
    for i, j, scale in ((0, 1, 1.0), (1, 2, 0.8), (0, 2, 0.3)):
        h_site[i, j] = h_site[j, i] = -float(e["hopping"]) * scale
    h = np.kron(h_site, np.eye(2))

    hamiltonian = np.einsum("ij,ijab->ab", h, bilinears).astype(complex)
    amplitudes = float(e["pairing"]) * _PAIR_SCALES * np.exp(
        1j * float(e["phase"]) * _PAIR_PHASES
    )
    for site, amplitude in enumerate(amplitudes):
        hamiltonian += amplitude * pair_creation[site]
        hamiltonian += amplitude.conjugate() * pair_annihilation[site]
    return hamiltonian


def _spectrum_data(e):
    """Return eigenenergies, transition energies, and spin weights.

    Rows represent final sample eigenstates and columns represent initial
    states. The returned weights do not include initial thermal probabilities
    or the detector spectral envelope.
    """

    _, _, _, spin = _fermion_operators()
    energies, vectors = eigh(_sample_hamiltonian(e), check_finite=False)
    operator = spin[int(e["site"])]
    matrix_elements = vectors.conj().T @ operator @ vectors
    transition_energies = energies[:, None] - energies[None, :]
    return energies, transition_energies, np.abs(matrix_elements) ** 2


def spectrum(e):
    """Return transition energies and spin-transition weights.

    Rows represent final sample eigenstates and columns represent initial
    states. The returned weights do not include initial thermal probabilities
    or the detector spectral envelope.
    """

    _, transition_energies, weights = _spectrum_data(e)
    return transition_energies, weights


def response(e):
    """Return the detector response with unit detector scale."""

    energies, transition_energies, weights = _spectrum_data(e)
    temperature = float(e["temperature"])

    # Subtracting the lowest energy prevents overflow in exp(-E/T).
    probabilities = np.exp(-(energies - energies.min()) / temperature)
    probabilities /= probabilities.sum()

    positive = transition_energies > 0.0
    frequency = transition_energies
    envelope = np.zeros_like(frequency)
    center = float(e["center"])
    width = float(e["width"])
    envelope[positive] = frequency[positive] ** 2 * np.exp(
        -0.5 * ((frequency[positive] - center) / width) ** 2
    )

    # The initial state is indexed by the second axis. The detector probes
    # supply the 2*pi*nu*|lambda|^2 factor in the stated spectral weight.
    return float(np.sum(weights * envelope * probabilities[None, :]))


def predict_at(experiments, gain):
    """Predict rates for a sequence of experiments at a fixed gain."""

    return float(gain) * np.asarray([response(experiment) for experiment in experiments])


class Model:
    def __init__(self):
        self.gain = None

    def fit(self, records):
        """Estimate the common detector scale from calibration records."""

        unit_responses = []
        values = []
        weights = []
        for record in records:
            sigma = float(record["sigma"])
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("each calibration sigma must be positive and finite")
            unit_responses.append(response(record["input"]))
            values.append(float(record["value"]))
            weights.append(1.0 / sigma**2)

        if not unit_responses:
            raise ValueError("fit requires at least one calibration record")

        x = np.asarray(unit_responses, dtype=float)
        y = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        denominator = np.sum(w * x * x)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("calibration responses do not identify the detector gain")

        estimate = np.sum(w * x * y) / denominator
        # The stated apparatus restricts the unknown scale to this interval.
        self.gain = float(np.clip(estimate, 0.6, 1.6))
        return self

    def predict(self, experiments):
        if self.gain is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.gain)
