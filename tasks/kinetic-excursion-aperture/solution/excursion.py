"""A first-return excursion with an interior position aperture and kick."""
import numpy as np
from scipy.special import erf, roots_legendre, ndtr, ndtri


from camera import free, middle


def alive(t, u, x, w, D, n=40):
    w = np.atleast_1d(w)
    nodes, weights = roots_legendre(n)
    y = (nodes + 1) / 2
    s = t * y[:, None]
    R = (np.sqrt(D * t) * y / (1 - y))[None, :]
    first = R * free(s, u, 0.0, -R, D) * erf(np.sqrt(3 * u * R / (D * s)))
    cut = (
        t
        / 4
        * weights[:, None]
        * weights[None, :]
        * np.sqrt(D * t)
        / (1 - y)[None, :] ** 2
        * first
    )
    after = free((t - s)[..., None], -R[..., None], x, w, D)
    return np.maximum(
        free(t, u, x, w, D) - np.sum(cut[..., None] * after, axis=(0, 1)), 0
    )


def query(q, D, n=36, unrestricted=()):
    u, v = q["launch"], q["incoming"]
    s, r = q["times"]
    x = q["position"]
    mode = q["mode"]
    # Signs here refer to reversed incoming velocities on the second killed leg.
    left = (
        free
        if "left" in unrestricted
        else lambda t, u, x, w, D: alive(t, u, x, w, D, n)
    )
    right = (
        free
        if "right" in unrestricted
        else lambda t, u, x, w, D: alive(t, u, x, w, D, n)
    )
    if mode == "clear":
        w = q["velocity"]
        return float(
            (
                left(s, u, x, w, D)
                / free(s, u, x, w, D)
                * right(r, v, x, -w, D)
                / free(r, v, x, -w, D)
            ).item()
        )
    # Unrestricted Gaussian product in the middle velocity. Integrate its
    # finite physical range using the analytic Gaussian conditional cdf.
    nodes, weights = roots_legendre(n)

    def joint(delta, threshold):
        # The curvature of log free_t(x,w|u) as a polynomial in w is -2/(D t).
        mean, std, weight = middle(u, v, s, r, x, delta, D)
        p0 = 0 if threshold is None else ndtr((threshold - mean) / std)
        prob = p0 + (1 - p0) * (nodes + 1) / 2
        w = mean + std * ndtri(prob)
        ratio = (
            left(s, u, x, w, D)
            / free(s, u, x, w, D)
            * right(r, v, x, -w - delta, D)
            / free(r, v, x, -w - delta, D)
        )
        return weight * (1 - p0) * np.dot(weights / 2, ratio)

    if mode == "direction":
        return joint(0, q["cut"]) / joint(0, None)
    plus = joint(q["kick"], q["cut"])
    minus = joint(-q["kick"], q["cut"])
    return plus / (plus + minus)
