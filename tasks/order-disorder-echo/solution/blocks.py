"""Ising order/disorder dipoles on a regulated cylinder."""
from itertools import product
import numpy as np


def response(q, steps=96, ignore_disorder=False, reset_sheet=False):
    marks = q["pulses"]
    n = len(marks)
    odd = q.get("sector", "even") == "odd"
    m = n + int(odd)
    left = np.array([p[1] for p in marks])
    right = np.array([p[2] for p in marks])
    dual = np.array([p[0] == "string" for p in marks])
    gap = np.array(q["segments"])
    height = np.r_[0.0, np.cumsum(gap[:, 0])]
    real = np.r_[0.0, np.cumsum(gap[:, 1])]
    t = np.linspace(0, 1, steps + 1)[:, None]
    parts = []
    signs = np.array([(1, *s) for s in product((1, -1), repeat=m - 1)])
    channels = np.array([p for p in product((0, 1), repeat=m) if sum(p) % 2 == 0])
    characters = (-1) ** (channels @ ((1 - signs) // 2).T)
    for chirality in (1, -1):
        L = left - chirality * t * real + 1j * chirality * height
        R = right - chirality * t * real + 1j * chirality * height
        powers = np.zeros((len(t), len(signs)), complex)
        for i in range(n):
            for j in range(i + 1, n):
                # Fixed sign of the imaginary separation selects a log branch
                # analytic throughout each real-time evolution.
                def logsin(z):
                    return np.log(1 - np.exp(-chirality * 2j * np.pi * z))

                u = logsin(L[:, i] - L[:, j]) + logsin(R[:, i] - R[:, j])
                u -= logsin(L[:, i] - R[:, j]) + logsin(R[:, i] - L[:, j])
                powers += u[:, None] * (signs[:, i] * signs[:, j]) / 4
        if odd:
            for j in range(n):
                powers += (
                    2j
                    * np.pi
                    * chirality
                    * (left[j] - right[j])
                    * (signs[:, -1] * signs[:, j])
                    / 4
                )
        parts.append(np.exp(powers) @ characters.T)
    squares = parts[0] * parts[1]
    roots = np.sqrt(squares)
    if not reset_sheet:
        # These square roots are analytic conformal blocks. The block can
        # cross zero, so continue its value and slope through a nearby zero.
        for j in range(1, len(roots)):
            guess = roots[j - 1] if j == 1 else 2 * roots[j - 1] - roots[j - 2]
            if j >= 3:
                guess = 3 * roots[j - 1] - 3 * roots[j - 2] + roots[j - 3]
            flip = np.real(guess.conj() * roots[j]) < 0
            roots[j, flip] *= -1
    channels_final = roots[-1]
    if odd:
        dual = np.r_[dual, False]
    weight = (
        np.ones(len(channels))
        if ignore_disorder
        else (-1) ** (channels @ dual.astype(int))
    )
    return (channels_final @ weight) / (2 ** (m - 1))
