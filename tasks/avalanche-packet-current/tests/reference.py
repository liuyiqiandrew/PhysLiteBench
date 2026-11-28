"""Evaluate the independent chronological history integrals in horizontal strips."""
import numpy as np
from itertools import combinations
from numpy.polynomial.legendre import leggauss


def coefficients(q, parents, pins):
    dim = len(parents) - sum(p is not None for p in pins)
    p = np.vstack([np.zeros(dim), np.eye(dim)]) if dim else np.empty((1, 0))
    one = np.ones(len(p))
    ve, vh = q["velocities"]
    vs = (ve, vh)
    dead = np.array(q["dead_space"]) / vs
    T = np.array([(1 - q["depth"]) / ve, q["depth"] / vh]) + q["last_delays"]
    start = [one * 0]
    pos = [one * q["depth"]]
    ready = [one * d for d in dead]
    end = [one * (1 - pos[0]) / ve, one * pos[0] / vh]
    packets = [[0], [0]]
    constraints = []
    k = 0
    for j, i in enumerate(parents):
        b = i // 2
        c = i % 2
        if j == pins[0]:
            t = (T[0] - (1 - pos[b]) / ve + (vh / ve) * start[b]) / (1 + vh / ve)
        elif j == pins[1]:
            t = (T[1] - pos[b] / vh + (ve / vh) * start[b]) / (1 + ve / vh)
        else:
            t = p[:, k]
            k += 1
        constraints.extend([t - start[-1], t - ready[i], end[i] - t])
        ready[i] = t + dead[c]
        ready.extend([t + dead[0], t + dead[1]])
        x = pos[b] + (ve if c == 0 else -vh) * (t - start[b])
        start.append(t)
        pos.append(x)
        end.extend([t + (1 - x) / ve, t + x / vh])
        packets[0].append(packets[0][b] if c == 0 else j + 1)
        packets[1].append(packets[1][b] if c == 1 else j + 1)
    for c, pin in enumerate(pins):
        if pin is not None:
            constraints.extend(
                [T[c] - e for e, r in zip(end[c::2], packets[c]) if r != pin + 1]
            )
    marks = q["read"] + (q["tag"][:2] if q["tag"] else [])
    cuts = [t - v for t in start + end for v in marks]

    def transform(rows):
        z = np.array(rows)
        return np.c_[z[:, 0], z[:, 1:] - z[:, :1]]

    return dim, min(T), transform(constraints), transform(cuts)


def integration_nodes(q, parents, pins, n):
    dim, T, ineq, cuts = coefficients(q, parents, pins)
    u, w = leggauss(n)
    u = (u + 1) / 2
    w = w / 2
    if dim == 0:
        return (
            (np.empty((1, 0)), np.ones(1))
            if min(ineq[:, 0]) >= -1e-11
            else (None, None)
        )
    vert = ineq if dim == 1 else ineq[abs(ineq[:, 2]) < 1e-10]
    lo, hi = 0.0, T
    for row in vert:
        a, b = row[:2]
        if abs(b) < 1e-11:
            if a < -1e-10:
                return None, None
        elif b > 0:
            lo = max(lo, -a / b)
        else:
            hi = min(hi, -a / b)
    if hi - lo < 1e-11:
        return None, None
    vcuts = cuts if dim == 1 else cuts[abs(cuts[:, 2]) < 1e-10]
    knots = [lo, hi] + [
        -s[0] / s[1] for s in vcuts if abs(s[1]) > 1e-11 and lo < -s[0] / s[1] < hi
    ]
    if dim == 2:
        lower = [[0.0, 0.0]]
        upper = [[T, 0.0]]
        markers = []
        for source, tag in [(ineq, False), (cuts, True)]:
            for a, b, c in source:
                if abs(c) > 1e-10:
                    (markers if tag else lower if c > 0 else upper).append(
                        [-a / c, -b / c]
                    )
        lower, upper, markers = [
            np.unique(np.round(a, 12), axis=0) if len(a) else np.empty((0, 2))
            for a in (lower, upper, markers)
        ]
        lines = np.vstack([lower, upper, markers])
        for a, b in combinations(lines, 2):
            slope = a[1] - b[1]
            if abs(slope) < 1e-11:
                continue
            x = (b[0] - a[0]) / slope
            if lo < x < hi:
                y = a[0] + a[1] * x
                if (
                    max(lower[:, 0] + x * lower[:, 1]) - 1e-9
                    <= y
                    <= min(upper[:, 0] + x * upper[:, 1]) + 1e-9
                ):
                    knots.append(x)
    knots = np.unique(np.round(knots, 12))
    starts = knots[:-1, None]
    width = np.diff(knots)[:, None]
    x = (starts + width * u).ravel()
    wx = (width * w).ravel()
    if dim == 1:
        return x[:, None], wx
    L = np.max(lower[:, 0] + x[:, None] * lower[:, 1], axis=1)
    H = np.min(upper[:, 0] + x[:, None] * upper[:, 1], axis=1)
    valid = H > L + 1e-11
    if not np.any(valid):
        return None, None
    x, wx, L, H = (a[valid] for a in (x, wx, L, H))
    mid = np.clip(markers[:, 0] + x[:, None] * markers[:, 1], L[:, None], H[:, None])
    ystops = np.sort(np.c_[L, mid, H], axis=1)
    dy = np.diff(ystops, axis=1)
    ix, j = np.where(dy > 1e-11)
    ys = ystops[ix, j, None] + dy[ix, j, None] * u
    xs = np.broadcast_to(x[ix, None], ys.shape)
    weights = wx[ix, None] * dy[ix, j, None] * w
    return np.c_[xs.ravel(), ys.ravel()], weights.ravel()


from itertools import product


def integrate_history(q, scale, parents, pins, u, weights, terminal=False):
    speed = np.array(q["velocities"])
    dead = np.array(q["dead_space"]) / speed
    rate = scale * np.array(q["ionization"]) * speed
    target = (np.array([1 - q["depth"], q["depth"]]) / speed) + q["last_delays"]
    N = len(weights)
    one = np.ones(N)
    zero = np.zeros(N)
    newborn = [zero]
    place = [one * q["depth"]]
    last = [zero, zero]
    finish = [zero + (1 - q["depth"]) / speed[0], zero + q["depth"] / speed[1]]
    ready_segments = []
    packet = [[0], [0]]
    density = one.copy()
    free = 0
    new_packet = [0 if pin is None else pin + 1 for pin in pins]
    for j, i in enumerate(parents):
        c = i % 2
        p = i // 2
        left = np.maximum(newborn[-1], last[i] + dead[c])
        right = finish[i]
        width = np.maximum(right - left, 0.0)
        if j == pins[0]:
            t = (
                target[0]
                - (1 - place[p]) / speed[0]
                + (speed[1] / speed[0]) * newborn[p]
            ) / (1 + speed[1] / speed[0])
            density *= ((t >= left) & (t < right)) / (1 + speed[1] / speed[0])
        elif j == pins[1]:
            t = (
                target[1] - place[p] / speed[1] + (speed[0] / speed[1]) * newborn[p]
            ) / (1 + speed[0] / speed[1])
            density *= ((t >= left) & (t < right)) / (1 + speed[0] / speed[1])
        else:
            t = u[:, free]
            free += 1
            density *= (t >= left - 1e-10) & (t <= right + 1e-10)
        x = place[p] + (1 if c == 0 else -1) * speed[c] * (t - newborn[p])
        x = np.clip(x, 0, 1)
        ready_segments.append((last[i] + dead[c], t, c))
        last[i] = t
        last.extend([t, t])
        finish.extend([t + (1 - x) / speed[0], t + x / speed[1]])
        newborn.append(t)
        place.append(x)
        packet[0].append(packet[0][p] if c == 0 else j + 1)
        packet[1].append(packet[1][p] if c == 1 else j + 1)
    for i, s in enumerate(last):
        ready_segments.append((s + dead[i % 2], finish[i], i % 2))
    multiplicity = tuple(pkt.count(origin) for pkt, origin in zip(packet, new_packet))
    for c, pinned in enumerate(pins):
        if pinned is not None:
            for p, origin in enumerate(packet[c]):
                if origin != new_packet[c]:
                    density *= finish[2 * p + c] < target[c]
    if not np.any(density):
        return multiplicity, 0.0, 0.0

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
    return multiplicity, density @ weights, (sample * density) @ weights


def packet_law(q, scale=1.0, order=36):
    n = q["gain"] - 1
    total = np.zeros((n + 2, n + 2))
    moment = np.zeros_like(total)
    for parents in product(*(range(2 + 2 * j) for j in range(n))):
        origins = [[j for j, i in enumerate(parents) if i % 2 != c] for c in (0, 1)]
        choices = [
            (o if tau > 0 else [] if o else [None])
            for tau, o in zip(q["last_delays"], origins)
        ]
        for pins in product(*choices):
            u, w = integration_nodes(q, parents, pins, order)
            if u is None:
                continue
            idx, a, b = integrate_history(q, scale, parents, pins, u, w)
            total[idx] += a
            moment[idx] += b
    return total, moment
