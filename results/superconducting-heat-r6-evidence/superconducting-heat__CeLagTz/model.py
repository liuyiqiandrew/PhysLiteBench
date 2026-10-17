"""Model for the phase-contrast heat-exchange apparatus.

All energies in the implementation are expressed in kelvin.  The factors at
the end of the rate integrals convert a conductance in microsiemens into pW.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit, ive


KB = 1.380649e-23
ELECTRON_CHARGE = 1.602176634e-19
HBAR = 1.054571817e-34

QUASIPARTICLE_DENSITY = 1.0e6       # states/K, per spin
NORMAL_POLARIZATION_TIME = 1.0e-5   # s
OSCILLATOR_ENERGY = 24.0            # hbar omega / kB, K

# G[us] * (kB/e)^2 gives watts after the appropriate 10^6 conversion.  The
# factor two is retained because the supplied nu includes both xi branches.
RATE_CONVERSION = 2.0e-6 * KB / ELECTRON_CHARGE**2
HEAT_CONVERSION = 2.0e6 * (KB / ELECTRON_CHARGE)**2


def gap(temperature, critical):
    """Induced gap, in kelvin, using the prescribed apparatus law."""
    temperature = float(temperature)
    return 1.764 * critical * np.tanh(
        1.74 * np.sqrt(critical / temperature - 1.0)
    )


def _spin_loss(temperature, delta, polarization):
    """Integral of f(E,mu)-f(E,-mu) from delta to infinity and derivative."""
    x = (polarization - delta) / temperature
    y = (-polarization - delta) / temperature
    value = temperature * (np.logaddexp(0.0, x) - np.logaddexp(0.0, y))
    derivative = expit(x) + expit(y)
    return value, derivative


def _sidebands(rho, temperature, count=24):
    """Thermal probabilities for oscillator energy changes ell*hbar*omega."""
    ell = np.arange(-count, count + 1, dtype=int)
    if rho == 0.0:
        return ell, (ell == 0).astype(float)

    nb = 1.0 / np.expm1(OSCILLATOR_ENERGY / temperature)
    argument = 2.0 * rho * np.sqrt(nb * (nb + 1.0))
    probabilities = ive(np.abs(ell), argument) * np.exp(
        -rho * (2.0 * nb + 1.0)
        + argument
        + ell * OSCILLATOR_ENERGY / (2.0 * temperature)
    )
    # The omitted tail is negligible for rho<=1; normalization also makes
    # numerical sums exact.
    probabilities /= probabilities.sum()
    return ell, probabilities


@lru_cache(maxsize=4096)
def _channels(left, right, phase, amplitude, drive, rho, oscillator_temperature,
              nodes=192, sideband_count=24):
    """Build quadrature channels for all allowed elementary transitions."""
    left = float(left)
    right = float(right)
    phase = float(phase)
    amplitude = float(amplitude)
    drive = float(drive)
    rho = float(rho)
    oscillator_temperature = float(oscillator_temperature)

    dl, dr = gap(left, 1.2), gap(right, 1.9)
    x, weights = leggauss(nodes)
    ell_values, probabilities = _sidebands(
        rho, oscillator_temperature, sideband_count
    )
    probability_by_ell = dict(zip(ell_values.tolist(), probabilities.tolist()))

    upper = np.sqrt(80.0 * max(left, right) + 40.0)
    z = (x + 1.0) * upper / 2.0
    theta = (x + 1.0) * np.pi / 4.0
    channel_parts = []

    harmonics = [0] if amplitude == 0.0 else [-1, 0, 1]
    for harmonic in harmonics:
        if harmonic == 0:
            transfer_spins = ((1, 1, 0.5), (-1, -1, 0.5))
            pair_spins = ((1, -1, 0.5), (-1, 1, 0.5))
        else:
            transfer_spins = ((-harmonic, harmonic, amplitude**2 / 2.0),)
            pair_spins = ((harmonic, harmonic, amplitude**2 / 2.0),)

        for ell in ell_values:
            forward = probability_by_ell[int(ell)]
            backward = probability_by_ell[int(-ell)]
            if forward == 0.0 and backward == 0.0:
                continue

            recoil = float(ell) * OSCILLATOR_ENERGY
            shift = harmonic * drive - recoil
            parity = -1.0 if int(ell) & 1 else 1.0

            # Transfer: E_R=E_L+shift.  The z^2 substitution removes the
            # square-root singularity at the larger gap edge.
            edge = max(dl, dr - shift)
            energy = edge + z * z
            other = energy + shift
            denominator = np.sqrt(
                (energy - dl) * (energy + dl)
                * (other - dr) * (other + dr)
            )
            # The two Bogoliubov amplitudes remain coherent within each
            # Fourier harmonic.  The singlet coherence has the opposite sign
            # for a spin-flip transfer harmonic.  A displacement gives the
            # relative D/D^dagger sign (-1)^ell.
            coherence = (-1.0 if harmonic == 0 else 1.0) * np.cos(phase) * parity
            numerator = energy * other + coherence * dl * dr
            measure = (
                weights * upper / 2.0 * 2.0 * z
                * numerator / denominator
            )
            for sl, sr, weight in transfer_spins:
                channel_parts.append((
                    energy, other, measure * weight, sl, sr, False,
                    harmonic * drive, recoil, forward, backward,
                ))

            # Pair creation is allowed when the supplied gate/oscillator
            # energy exceeds the two gaps.
            if shift > dl + dr:
                available = shift - dl - dr
                energy_pair = dl + available * np.sin(theta) ** 2
                other_pair = shift - energy_pair
                pair_coherence = -np.cos(phase) * parity
                pair_numerator = energy_pair * other_pair + pair_coherence * dl * dr
                pair_measure = (
                    weights * np.pi / 4.0 * 2.0 * pair_numerator
                    / np.sqrt((energy_pair + dl) * (other_pair + dr))
                )
                for sl, sr, weight in pair_spins:
                    channel_parts.append((
                        energy_pair, other_pair, pair_measure * weight,
                        sl, sr, True, harmonic * drive, recoil,
                        forward, backward,
                    ))

    fields = list(zip(*channel_parts))
    if not fields:
        raise RuntimeError("No transition channels were generated")
    energy = np.stack(fields[0])
    other = np.stack(fields[1])
    measure = np.stack(fields[2])
    sl = np.asarray(fields[3], dtype=float)[:, None]
    sr = np.asarray(fields[4], dtype=float)[:, None]
    creation = np.asarray(fields[5], dtype=bool)[:, None]
    gate_work = np.asarray(fields[6], dtype=float)
    recoil = np.asarray(fields[7], dtype=float)
    forward = np.asarray(fields[8], dtype=float)[:, None]
    backward = np.asarray(fields[9], dtype=float)[:, None]
    injection = np.column_stack((
        np.where(creation[:, 0], sl[:, 0], -sl[:, 0]), sr[:, 0]
    ))
    return dl, dr, (
        energy, other, measure, sl, sr, creation, gate_work, recoil,
        forward, backward, injection,
    )


def _transition(left, right, oscillator_temperature, channels, polarization):
    """Return energy, spin, and derivative integrals for one spin state."""
    (energy, other, measure, sl, sr, creation, gate_work, recoil,
     forward, backward, injection) = channels

    fl = expit((sl * polarization[0] - energy) / left)
    fr = expit((sr * polarization[1] - other) / right)
    dfl = fl * (1.0 - fl) / left
    dfr = fr * (1.0 - fr) / right

    transfer_occupation = forward * fl * (1.0 - fr) - backward * fr * (1.0 - fl)
    pair_occupation = forward * (1.0 - fl) * (1.0 - fr) - backward * fl * fr
    occupation = np.where(creation, pair_occupation, transfer_occupation)

    dmu_l = np.where(
        creation,
        -sl * dfl * (forward * (1.0 - fr) + backward * fr),
        sl * dfl * (forward * (1.0 - fr) + backward * fr),
    )
    dmu_r = -sr * dfr * np.where(
        creation,
        forward * (1.0 - fl) + backward * fl,
        forward * fl + backward * (1.0 - fl),
    )

    weighted = measure * occupation
    net = weighted.sum(axis=1)
    heat = np.array((
        np.sum(weighted * np.where(creation, -energy, energy)),
        -np.sum(weighted * other),
        np.dot(gate_work, net),
        np.dot(recoil, net),
    ))
    spin = injection.T @ net
    jacobian = injection.T @ np.column_stack((
        np.sum(measure * dmu_l, axis=1),
        np.sum(measure * dmu_r, axis=1),
    ))
    affinity = np.where(
        creation,
        (energy - sl * polarization[0]) / left,
        -(energy - sl * polarization[0]) / left,
    ) + (other - sr * polarization[1]) / right + recoil[:, None] / oscillator_temperature
    entropy = np.sum(weighted * affinity)
    return heat, spin, jacobian, entropy


@lru_cache(maxsize=4096)
def _stationary(left, right, phase, amplitude, drive, rho,
                oscillator_temperature, conductance):
    """Solve the two stationary spin-number balance equations."""
    ratio = QUASIPARTICLE_DENSITY / (
        NORMAL_POLARIZATION_TIME * RATE_CONVERSION * conductance
    )
    bound = 20.0
    guess = np.zeros(2)

    for _ in range(10):
        dl, dr, channels = _channels(
            left, right, phase, amplitude, drive, rho,
            oscillator_temperature,
        )

        def residual(mu):
            spin = _transition(left, right, oscillator_temperature, channels, mu)[1]
            losses = np.array((
                _spin_loss(left, dl, mu[0])[0],
                _spin_loss(right, dr, mu[1])[0],
            ))
            return spin - ratio * losses

        def derivative(mu):
            jac = _transition(left, right, oscillator_temperature, channels, mu)[2]
            losses = np.array((
                _spin_loss(left, dl, mu[0])[1],
                _spin_loss(right, dr, mu[1])[1],
            ))
            return jac - ratio * np.diag(losses)

        result = least_squares(
            residual, guess, jac=derivative, bounds=(-bound, bound),
            x_scale="jac", ftol=1e-13, xtol=1e-13, gtol=1e-13,
            max_nfev=300,
        )
        mu = result.x
        for _ in range(20):
            value = residual(mu)
            if np.max(np.abs(value)) < 1e-11:
                break
            correction = np.linalg.solve(derivative(mu), value)
            step = 1.0
            old_norm = np.max(np.abs(value))
            while step > 1e-8:
                candidate = mu - step * correction
                if np.max(np.abs(candidate)) < bound:
                    new_norm = np.max(np.abs(residual(candidate)))
                    if new_norm < old_norm:
                        mu = candidate
                        break
                step *= 0.5
            if step <= 1e-8:
                break

        final_residual = residual(mu)
        if np.max(np.abs(final_residual)) < 1e-8 and np.max(np.abs(mu)) < 0.8 * bound:
            heat, spin, jacobian, entropy = _transition(
                left, right, oscillator_temperature, channels, mu
            )
            losses = np.array((
                _spin_loss(left, dl, mu[0])[0],
                _spin_loss(right, dr, mu[1])[0],
            ))
            return {
                "polarization": mu,
                "heat": HEAT_CONVERSION * conductance * heat,
                "spin": RATE_CONVERSION * conductance * spin,
                "residual": final_residual,
                "jacobian": derivative(mu),
                "entropy": RATE_CONVERSION * conductance * KB * entropy,
                "flip_entropy": RATE_CONVERSION * conductance * KB * ratio * np.sum(
                    mu * losses / np.array((left, right))
                ),
            }
        bound *= 2.0
        guess = mu

    raise RuntimeError("Stationary spin balance did not converge")


def _wrap_phase(phase):
    return (float(phase) + np.pi) % (2.0 * np.pi) - np.pi


def _mean_heat(experiment, conductance):
    state = _stationary(
        float(experiment["left_temperature"]),
        float(experiment["right_temperature"]),
        float(experiment["phase"]),
        float(experiment.get("amplitude", 0.0)),
        float(experiment.get("drive_energy", 0.3)),
        float(experiment.get("oscillator_coupling", 0.0)),
        float(experiment.get("oscillator_temperature", 12.0)),
        float(conductance),
    )
    return float(state["heat"][0])


def predict_at(experiments, conductance):
    values = []
    for experiment in experiments:
        statistic = experiment.get("statistic", "mean")
        mean = _mean_heat(experiment, conductance)
        if statistic == "mean":
            values.append(mean)
        elif statistic == "phase_contrast":
            shifted = dict(experiment, phase=_wrap_phase(
                experiment["phase"] + np.pi
            ))
            values.append((mean - _mean_heat(shifted, conductance)) / 2.0)
        else:
            raise ValueError("statistic must be 'mean' or 'phase_contrast'")
    return np.asarray(values, dtype=float)


class Model:
    def __init__(self):
        self.conductance = None

    def fit(self, records):
        """Fit the single conductance parameter by weighted least squares."""
        if not records:
            raise ValueError("at least one calibration record is required")
        target = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if (not np.isfinite(target).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError("calibration values and sigmas must be finite, positive numbers")
        experiments = [record["input"] for record in records]

        def objective(conductance):
            prediction = predict_at(experiments, float(conductance))
            residual = (prediction - target) / sigma
            return float(residual @ residual)

        result = minimize_scalar(
            objective, bounds=(20.0, 80.0), method="bounded",
            options={"xatol": 1e-9, "maxiter": 100},
        )
        candidates = [
            (result.fun, result.x),
            (objective(20.0), 20.0),
            (objective(80.0), 80.0),
        ]
        _, best = min(candidates, key=lambda item: item[0])
        self.conductance = float(best)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.conductance)


__all__ = ["Model", "gap", "predict_at"]
