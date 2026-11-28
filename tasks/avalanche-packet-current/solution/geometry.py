"""Enumerate laboratory-time polytopes of physical carrier histories."""
from itertools import product
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(None)
def rule(n):
    x, w = leggauss(n)
    return (x + 1) / 2, w / 2


def clip(poly, plane):
    if not len(poly):
        return poly
    v = plane[0] + poly @ plane[1:]
    if np.min(v) >= -1e-13:
        return poly
    if np.max(v) <= 1e-13:
        return np.empty((0, 2))
    out = []
    for j, p in enumerate(poly):
        prev = poly[j - 1]
        a, b = v[j - 1], v[j]
        if (a >= 0) != (b >= 0):
            out.append(prev + a / (a - b) * (p - prev))
        if b >= 0:
            out.append(p)
    return np.array(out)


def region_nodes(constraints, cuts, dim, T, n):
    if dim == 0:
        return (
            (np.ones((1, 1)), np.ones(1))
            if all(c[0] >= -1e-12 for c in constraints)
            else (None, None)
        )
    x, w = rule(n)
    if dim == 1:
        lo, hi = 0.0, T
        for a, b in constraints:
            if abs(b) < 1e-12:
                if a < -1e-12:
                    return None, None
            elif b > 0:
                lo = max(lo, -a / b)
            else:
                hi = min(hi, -a / b)
        if hi <= lo + 1e-13:
            return None, None
        k = sorted(
            set(
                [lo, hi]
                + [-a / b for a, b in cuts if abs(b) > 1e-12 and lo < -a / b < hi]
            )
        )
        u = np.concatenate([a + (b - a) * x for a, b in zip(k, k[1:])])
        weight = np.concatenate([(b - a) * w for a, b in zip(k, k[1:])])
        return np.c_[np.ones(len(u)), u], weight
    region = np.array([[0.0, 0.0], [T, 0.0], [T, T], [0.0, T]])
    for c in constraints:
        region = clip(region, c)
    if len(region) < 3:
        return None, None
    cells = [region]
    for line in cuts:
        split = []
        for poly in cells:
            v = line[0] + poly @ line[1:]
            if min(v) < -5e-11 and max(v) > 5e-11:
                split.extend([clip(poly, line), clip(poly, -line)])
            else:
                split.append(poly)
        cells = split
    X, Y = np.meshgrid(x, x, indexing="ij")
    U = X.ravel()
    V = ((1 - X) * Y).ravel()
    weights = (np.outer(w, w) * (1 - X)).ravel()
    nodes = []
    ws = []
    for poly in cells:
        for i in range(2, len(poly)):
            a, b, c = poly[0], poly[i - 1], poly[i]
            jac = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if jac > 1e-15:
                nodes.append(a[None, :] + U[:, None] * (b - a) + V[:, None] * (c - a))
                ws.append(jac * weights)
    if not nodes:
        return None, None
    t = np.concatenate(nodes)
    return np.c_[np.ones(len(t)), t], np.concatenate(ws)


def packet_law(q, scale=1.0, order=8, terminal=False):
    ve, vh = q["velocities"]
    vs = (ve, vh)
    dead = np.asarray(q["dead_space"]) / vs
    rates = np.asarray(q["ionization"]) * vs * scale
    n = q["gain"] - 1
    delays = q["last_delays"]
    target_times = np.array(
        [(1 - q["depth"]) / ve + delays[0], q["depth"] / vh + delays[1]]
    )
    dim = n - sum(x > 0 for x in delays)
    basis = np.eye(dim + 1)
    one = basis[0]
    zero = 0 * one
    T = min(target_times)
    total = np.zeros((n + 2, n + 2))
    score = np.zeros_like(total)
    for labels in product(*(range(2 * j + 2) for j in range(n))):
        packets = [[0], [0]]
        origins = [[], []]
        for j, i in enumerate(labels):
            parent = i % 2
            for c in (0, 1):
                packets[c].append(packets[c][i // 2] if c == parent else j + 1)
                if c != parent:
                    origins[c].append(j)
        choices = [
            (o if delays[c] > 0 else [] if o else [None]) for c, o in enumerate(origins)
        ]
        for pins in product(*choices):
            roots = [0 if j is None else j + 1 for j in pins]
            multiplicity = tuple(p.count(r) for p, r in zip(packets, roots))
            births = [zero]
            positions = [q["depth"] * one]
            ends = [(one - positions[0]) / ve, positions[0] / vh]
            fresh = [zero, zero]
            tracks = []
            bounds = []
            k = 1
            for j, i in enumerate(labels):
                p = i // 2
                c = i % 2
                born, x = births[p], positions[p]
                if j == pins[0]:
                    t = (ve * target_times[0] * one - one + x + vh * born) / (ve + vh)
                elif j == pins[1]:
                    t = (vh * target_times[1] * one - x + ve * born) / (ve + vh)
                else:
                    t = basis[k]
                    k += 1
                bounds.extend(
                    [t - births[-1], t - fresh[i] - dead[c] * one, ends[i] - t]
                )
                tracks.append((fresh[i], t, c))
                fresh[i] = t
                fresh.extend([t, t])
                xnew = x + (ve if c == 0 else -vh) * (t - born)
                births.append(t)
                positions.append(xnew)
                ends.extend([t + (one - xnew) / ve, t + xnew / vh])
            for i, b in enumerate(fresh):
                tracks.append((b, ends[i], i % 2))
            for c, origin in enumerate(roots):
                if pins[c] is not None:
                    bounds.extend(
                        [
                            target_times[c] * one - ends[2 * p + c]
                            for p, r in enumerate(packets[c])
                            if r != origin
                        ]
                    )
            gates = set(q["read"] + (q["tag"][:2] if q["tag"] else []))
            cuts = [e - g * one for e in [*births, *ends] for g in gates]
            for a, b, c in tracks:
                cuts.append(b - a - dead[c] * one)
            t, w = region_nodes(bounds, cuts, dim, T, order)
            if t is None:
                continue
            w = w * np.prod([vs[c] / (ve + vh) for c in (0, 1) if pins[c] is not None])
            times = t @ np.array(births).T
            contacts = t @ np.array(ends).T
            start = t @ np.array([a for a, b, c in tracks]).T
            end = t @ np.array([b for a, b, c in tracks]).T
            types = [c for a, b, c in tracks]

            def Q(z):
                res = np.zeros_like(z, dtype=float)
                for j, born in enumerate(times.T):
                    for c, v in enumerate(vs):
                        res += v * np.maximum(
                            0,
                            np.minimum(contacts[:, 2 * j + c, None], z) - born[:, None],
                        )
                return res

            cutoff = np.minimum(end, start + dead[types])
            events = np.concatenate([times, contacts, cutoff, end], axis=1)
            perm = np.argsort(events, axis=1)
            sorted_t = np.take_along_axis(events, perm, axis=1)
            C = Q(sorted_t)
            eta = q["feedback"]
            s = -eta * np.diff(C, axis=1)
            ratio = np.ones_like(s)
            np.divide(np.expm1(s), s, out=ratio, where=abs(s) > 1e-13)
            H = np.c_[
                np.zeros(len(t)),
                np.cumsum(
                    np.diff(sorted_t, axis=1) * np.exp(-eta * C[:, :-1]) * ratio, axis=1
                ),
            ]
            H = np.take_along_axis(H, np.argsort(perm, axis=1), axis=1)
            count = len(tracks)
            cum = (H[:, -count:] - H[:, -2 * count : -count]) @ rates[types]
            likelihood = np.exp(-cum)
            if n:
                likelihood *= np.prod(rates[[i % 2 for i in labels]]) * np.exp(
                    -eta * Q(times[:, 1:]).sum(axis=1)
                )

            def window(data):
                z = np.broadcast_to(np.asarray(data[:2]), (len(t), 2))
                charge = (
                    (contacts[:, ::2, None] <= z[:, None, :]).sum(axis=1)
                    if terminal
                    else Q(z)
                )
                return charge[:, 1] - charge[:, 0]

            if q["tag"]:
                lo, hi, z, sigma = q["tag"]
                likelihood *= np.exp(-0.5 * ((window(q["tag"]) - z) / sigma) ** 2)
            mass = w * likelihood
            total[multiplicity] += mass.sum()
            score[multiplicity] += mass @ (window(q["read"]) / q["gain"])
    return total, score


def observable(q, law, independent=False):
    mass, weighted = law
    m, h = (slice(None) if n is None else n for n in q["multiplicities"])
    selected = mass[m, h].sum()
    norm = mass.sum()
    if q["observable"] == "waveform":
        return weighted[m, h].sum() / selected
    if independent:
        return mass[m, :].sum() * mass[:, h].sum() / (norm * norm)
    return selected / norm
