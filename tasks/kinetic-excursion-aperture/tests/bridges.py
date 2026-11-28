import numpy as np
from scipy.special import ndtr, ndtri


def midstate(t, u, x, w, D, rng):
    # A half step of integrated Brownian motion, conditioned on both endpoint states.
    h = t / 2
    P = np.array([[1, h], [0, 1]])
    Q = 2 * D * np.array([[h**3 / 3, h * h / 2], [h * h / 2, h]])
    gain = Q @ P.T @ np.linalg.inv(P @ Q @ P.T + Q)
    cov = Q - gain @ P @ Q
    noise = np.linalg.cholesky(cov) @ rng.normal(size=(2, len(u)))
    predx = h * u
    return (
        predx + gain[0, 0] * (x - t * u) + gain[0, 1] * (w - u) + noise[0],
        u + gain[1, 0] * (x - t * u) + gain[1, 1] * (w - u) + noise[1],
    )


def mask(t, u, x, w, D, depth, rng):
    b = len(w)
    pos = np.stack([np.zeros(b), np.full(b, x)])
    vel = np.stack([np.full(b, u), w])
    keep = np.ones(b, bool)
    for k in range(depth):
        # Retain the same Brownian endpoints and reveal exact conditional midpoints.
        m, v = midstate(
            t / 2**k,
            vel[:-1].ravel(),
            np.diff(pos, axis=0).ravel(),
            vel[1:].ravel(),
            D,
            rng,
        )
        z = m.reshape(len(pos) - 1, b) + pos[:-1]
        keep &= (z > 0).all(axis=0)
        nm = 2 * len(pos) - 1
        newpos = np.empty((nm, b))
        newvel = np.empty_like(newpos)
        newpos[::2] = pos
        newpos[1::2] = z
        newvel[::2] = vel
        newvel[1::2] = v.reshape(len(pos) - 1, b)
        pos, vel = newpos, newvel
    return keep


def sample(q, D, n=2**15, depth=6, seed=892):
    rng = np.random.default_rng(seed)
    u, v = q["launch"], q["incoming"]
    s, r = q["times"]
    x = q["position"]

    def pair(delta, gate):
        hits = 0
        for i in range(0, n, 1024):
            b = min(n - i, 1024)
            if q["mode"] == "clear":
                w = np.full(b, q["velocity"])
                weight = 1.0
            else:
                npred = np.array([u * s, u])
                nstate = 2 * D * np.array([[s**3 / 3, s * s / 2], [s * s / 2, s]])
                cb = nstate[1, 0] / nstate[0, 0]
                before = u + cb * (x - npred[0])
                prior = nstate[1, 1] - cb * nstate[0, 1]
                final = np.array([0.0, -v])
                pivot = np.array([x + (before + delta) * r, before + delta])
                slope = np.array([r, 1.0])
                cov = 2 * D * np.array(
                    [[r**3 / 3, r * r / 2], [r * r / 2, r]]
                ) + prior * np.outer(slope, slope)
                k = prior * np.linalg.solve(cov, slope)
                mean = before + k @ (final - pivot)
                std = np.sqrt(prior - prior * (k @ slope))
                p = 0 if gate is None else ndtr((gate - mean) / std)
                w = mean + std * ndtri(p + (1 - p) * rng.random(b))
                weight = (
                    np.exp(-0.5 * (final - pivot) @ np.linalg.solve(cov, final - pivot))
                    * (1 - p)
                    / np.sqrt(np.linalg.det(cov))
                )
            good = mask(s, u, x, w, D, depth, rng) & mask(
                r, v, x, -w - delta, D, depth, rng
            )
            hits += np.sum(good)
        return weight * hits / n

    if q["mode"] == "clear":
        return pair(0, None)
    if q["mode"] == "direction":
        return pair(0, q["cut"]) / pair(0, None)
    a = pair(q["kick"], q["cut"])
    b = pair(-q["kick"], q["cut"])
    return a / (a + b)
