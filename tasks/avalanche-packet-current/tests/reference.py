"""Joint chronological ionization density and gated displacement current for a finite-gain avalanche."""
from itertools import product
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(None)
def quadrature(dim, order):
    if not dim:
        return np.zeros((1, 0)), np.ones(1)
    x, w = leggauss(order)
    coords = np.meshgrid(*([(1 + x) / 2] * dim), indexing="ij")
    weights = np.meshgrid(*([w / 2] * dim), indexing="ij")
    return np.stack(coords, axis=-1).reshape(-1, dim), np.prod(weights, axis=0).ravel()


def integrate_history(q, scale, parents, hit, u, weights, terminal=False):
    speed = np.array(q["velocities"])
    dead = np.array(q["dead_space"]) / speed
    rate = scale * np.array(q["ionization"]) * speed
    target = (1 - q["depth"]) / speed[0] + q["last_delay"]
    N = len(weights)
    one = np.ones(N)
    zero = np.zeros(N)
    newborn = [zero]
    place = [one * q["depth"]]
    last = [zero, zero]
    finish = [zero + (1 - q["depth"]) / speed[0], zero + q["depth"] / speed[1]]
    ready_segments = []
    packet = [0]
    density = one.copy()
    free = 0
    new_packet = 0 if hit is None else hit + 1
    for j, i in enumerate(parents):
        c = i % 2
        p = i // 2
        left = np.maximum(newborn[-1], last[i] + dead[c])
        right = finish[i]
        width = np.maximum(right - left, 0.0)
        if j == hit:
            t = (
                target - (1 - place[p]) / speed[0] + (speed[1] / speed[0]) * newborn[p]
            ) / (1 + speed[1] / speed[0])
            density *= ((t >= left) & (t < right)) / (1 + speed[1] / speed[0])
        else:
            t = left + u[:, free] * width
            free += 1
            density *= width
        x = place[p] + (1 if c == 0 else -1) * speed[c] * (t - newborn[p])
        x = np.clip(x, 0, 1)
        ready_segments.append((last[i] + dead[c], t, c))
        last[i] = t
        last.extend([t, t])
        finish.extend([t + (1 - x) / speed[0], t + x / speed[1]])
        newborn.append(t)
        place.append(x)
        packet.append(packet[p] if c == 0 else j + 1)
    for i, s in enumerate(last):
        ready_segments.append((s + dead[i % 2], finish[i], i % 2))
    if q["multiplicity"] is not None and packet.count(new_packet) != q["multiplicity"]:
        return 0.0, 0.0
    if hit is not None:
        for p, origin in enumerate(packet):
            if origin != new_packet:
                density *= finish[2 * p] < target
    if not np.any(density):
        return 0.0, 0.0

    def charge(t):
        result = np.zeros_like(t)
        t = np.asarray(t)
        for birth, x in zip(newborn, place):
            b = birth if t.ndim == 1 else birth[:, None]
            pos = x if t.ndim == 1 else x[:, None]
            age = np.maximum(t - b, 0.0)
            result += np.minimum(speed[0] * age, 1 - pos) + np.minimum(
                speed[1] * age, pos
            )
        return result

    # Current, eligibility and collection change only at these breakpoints.
    breaks = np.sort(
        np.stack([*newborn, *finish, *[s for s, e, c in ready_segments]], axis=1),
        axis=1,
    )
    Q = charge(breaks)
    dq = np.diff(Q, axis=1)
    left = breaks[:, :-1]
    dt = np.diff(breaks, axis=1)
    mid = left + dt / 2
    base = np.zeros_like(mid)
    for start, end, c in ready_segments:
        base += rate[c] * ((mid >= start[:, None]) & (mid < end[:, None]))
    eta = q["feedback"]
    h = eta * dq
    phi = np.ones_like(h)
    np.divide(-np.expm1(-h), h, out=phi, where=h != 0)
    cum = (base * dt * np.exp(-eta * Q[:, :-1]) * phi).sum(axis=1)
    density *= np.exp(-cum)
    for j, i in enumerate(parents):
        density *= rate[i % 2] * np.exp(-eta * charge(newborn[j + 1]))

    def signal(t):
        if terminal:
            return sum((z <= t).astype(float) for z in finish[::2])
        return charge(one * t)

    if q["tag"] is not None:
        lo, hi, z, w = q["tag"]
        y = signal(hi) - signal(lo)
        density *= np.exp(-0.5 * ((y - z) / w) ** 2)
    lo, hi = q["read"]
    sample = (signal(hi) - signal(lo)) / q["gain"]
    return density @ weights, (sample * density) @ weights


def predict(q, scale=1.0, order=14, terminal=False):
    n = q["gain"] - 1
    total = moment = 0.0
    for path in product(*(range(2 + 2 * j) for j in range(n))):
        holes = [i for i, p in enumerate(path) if p % 2]
        candidates = ([None] if not holes else []) if q["last_delay"] == 0 else holes
        for hit in candidates:
            u, w = quadrature(n - int(hit is not None), order)
            a, b = integrate_history(q, scale, path, hit, u, w, terminal)
            total += a
            moment += b
    return moment / total, total
