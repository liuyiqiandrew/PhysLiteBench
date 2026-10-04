import numpy as np
from scipy.linalg import eigh


# There are six spin orbitals, hence only 2**6 many-body states.  Building
# these small operators once makes each prediction an exact diagonalization
# of the specified quadratic Hamiltonian, without assumptions about particle
# number or parity.
_N_MODES = 6
_HILBERT_DIM = 1 << _N_MODES


def _annihilation_operator(mode):
    operator = np.zeros((_HILBERT_DIM, _HILBERT_DIM), dtype=complex)
    lower_modes = (1 << mode) - 1
    for state in range(_HILBERT_DIM):
        if state & (1 << mode):
            sign = -1 if (state & lower_modes).bit_count() % 2 else 1
            operator[state ^ (1 << mode), state] = sign
    return operator


_ANNIHILATION = tuple(_annihilation_operator(mode)
                       for mode in range(_N_MODES))
_CREATION = tuple(operator.conj().T for operator in _ANNIHILATION)
_ONE_BODY = tuple(
    tuple(_CREATION[row] @ _ANNIHILATION[col] for col in range(_N_MODES))
    for row in range(_N_MODES)
)
_PAIR_CREATION = tuple(
    _CREATION[2 * site] @ _CREATION[2 * site + 1]
    for site in range(3)
)
_PAIR_ANNIHILATION = tuple(operator.conj().T for operator in _PAIR_CREATION)
_SPIN_Z = tuple(
    (_ONE_BODY[2 * site][2 * site] - _ONE_BODY[2 * site + 1][2 * site + 1]) / 2
    for site in range(3)
)


def energy_matrix(e):
    """Return the exact many-body Hamiltonian for one experiment."""
    h = np.diag(np.asarray([-.35, .15, .70], dtype=float) + e['offset'])
    for i, j, multiplier in ((0, 1, 1.0), (1, 2, .8), (0, 2, .3)):
        h[i, j] = h[j, i] = -e['hopping'] * multiplier
    one_body = np.kron(h, np.eye(2))

    hamiltonian = np.zeros((_HILBERT_DIM, _HILBERT_DIM), dtype=complex)
    for row in range(_N_MODES):
        for col in range(_N_MODES):
            hamiltonian += one_body[row, col] * _ONE_BODY[row][col]

    amplitudes = ((1.0, 0.0), (.85, 1.0), (1.15, -.4))
    for site, (d, b) in enumerate(amplitudes):
        delta = e['pairing'] * d * np.exp(1j * e['phase'] * b)
        hamiltonian += (delta * _PAIR_CREATION[site] +
                        delta.conjugate() * _PAIR_ANNIHILATION[site])
    return hamiltonian


def spectrum(e):
    """Return transition energies and equilibrium spin-noise weights.

    The first index labels the final sample eigenstate and the second labels
    the initial state.  Positive frequencies therefore correspond to energy
    gained by the sample when the probe de-excites.
    """
    energies, vectors = eigh(energy_matrix(e))
    thermal = np.exp(-(energies - energies.min()) / e['temperature'])
    thermal /= thermal.sum()
    matrix_element = vectors.conj().T @ _SPIN_Z[e['site']] @ vectors
    weights = np.abs(matrix_element) ** 2 * thermal[None, :]
    return energies[:, None] - energies[None, :], weights


def response(e):
    frequency, weight = spectrum(e)
    band = np.where(
        frequency > 0.0,
        frequency ** 2 * np.exp(-0.5 * ((frequency - e['center']) / e['width']) ** 2),
        0.0,
    )
    return float(np.sum(weight * band))


def predict_at(experiments, gain):
    return gain * np.asarray([response(experiment) for experiment in experiments],
                             dtype=float)


class Model:
    def __init__(self):
        self.gain = None

    def fit(self, records):
        """Estimate the common detector scale using weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        calculated = np.asarray(
            [response(record['input']) for record in records], dtype=float
        )
        measured = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(calculated).all() or
                not np.isfinite(measured).all() or
                not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            raise ValueError(
                "calibration values and uncertainties must be finite, with sigma > 0"
            )

        denominator = np.sum((calculated / sigma) ** 2)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("calibration records contain no usable response")
        estimate = np.sum(calculated * measured / sigma ** 2) / denominator

        # The apparatus specification constrains the unknown scale to this
        # interval; clipping also keeps predictions physical for bad data.
        self.gain = float(np.clip(estimate, .6, 1.6))
        return self

    def predict(self, experiments):
        if self.gain is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.gain)
