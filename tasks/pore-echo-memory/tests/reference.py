"""Propagate each site's complex magnetization on a Fourier grid."""
import numpy as np

TRUE_WIDTH = 1.073


def magnetization(scan, phases):
    amplitude = np.zeros_like(phases)
    amplitude[..., scan["start"]] = 1
    for i, sign in enumerate(scan["signs"]):
        if i:
            p = scan["hops"][i - 1]
            neighbors = np.roll(amplitude, 1, axis=-1) + np.roll(amplitude, -1, axis=-1)
            amplitude = (1 - p) * amplitude + (p / 2) * neighbors
        amplitude *= phases if sign == 1 else np.conj(phases)
    return amplitude.sum(axis=-1)


def integrate(values, charge, scale, curvature=False):
    # Site 3 supplies the common phase. Its Fourier index follows by total charge.
    coefficients = np.fft.fftn(values) / values.size
    modes = np.rint(np.fft.fftfreq(values.shape[0]) * values.shape[0])
    i, j, k = np.meshgrid(modes, modes, modes, indexing="ij", sparse=True)
    norm = i**2 + j**2 + k**2 + (charge - i - j - k)**2
    kernel = scale**2 * norm if curvature else np.exp(-0.5 * scale**2 * norm)
    answer = np.sum(coefficients * kernel)
    assert abs(answer.imag) < 1e-9
    return float(answer.real)


def evaluate(scan, scale, partner=None, curvature=False):
    # N samples per axis resolve all Fourier modes of the product exactly.
    count = len(scan["signs"]) + (len(partner["signs"]) if partner else 0)
    n = 2 * count + 1
    unit = np.exp(2j * np.pi * np.arange(n) / n)
    phases = np.ones((n, n, n, 4), dtype=complex)
    for axis, grid in enumerate(np.meshgrid(unit, unit, unit, indexing="ij", sparse=True)):
        phases[..., axis] = grid
    first = magnetization(scan, phases)
    q = sum(scan["signs"])
    echo = integrate(first, q, scale, curvature)
    if partner is None:
        return echo
    second = magnetization(partner, phases)
    s = sum(partner["signs"])
    joint = integrate(first * second.conjugate(), q - s, scale)
    return joint - echo * integrate(second, s, scale)
