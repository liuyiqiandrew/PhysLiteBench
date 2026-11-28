"""Integrate either the constant plateaus or affine level surfaces of the induced charge."""
import numpy as np
from itertools import combinations
from transport import clip, rule, packet_law


def spectrum_nodes(constraints, cuts, dim, T, order, forms, spec):
    target = spec["charge"]
    line = spec["part"] == "line"
    x, w = rule(order)
    samples = []
    weights = []

    def match(form):
        return np.linalg.norm(form[1:]) < 2e-10 and abs(form[0] - target) < 2e-9

    if dim == 0:
        if (
            not all(c[0] >= -1e-12 for c in constraints)
            or not line
            or not match(forms(np.ones(1)))
        ):
            return None, None
        return np.ones((1, 1)), np.ones(1)
    if dim == 1:
        lo, hi = 0.0, T
        for a, b in constraints:
            if abs(b) < 1e-12:
                if a < -1e-11:
                    return None, None
            elif b > 0:
                lo = max(lo, -a / b)
            else:
                hi = min(hi, -a / b)
        if hi - lo <= 1e-12:
            return None, None
        breaks = sorted(
            set(
                [lo, hi]
                + [-a / b for a, b in cuts if abs(b) > 1e-12 and lo < -a / b < hi]
            )
        )
        for a, b in zip(breaks, breaks[1:]):
            f = forms(np.array([1.0, (a + b) / 2]))
            s = f[1]
            if line and match(f):
                samples.extend(a + (b - a) * x)
                weights.extend(w * (b - a))
            elif not line and abs(s) > 2e-10:
                root = (target - f[0]) / s
                if a < root < b:
                    samples.append(root)
                    weights.append(1 / abs(s))
        if not samples:
            return None, None
        return np.c_[np.ones(len(samples)), samples], np.array(weights)
    region = np.array([[0.0, 0.0], [T, 0.0], [T, T], [0.0, T]])
    for p in constraints:
        region = clip(region, p)
    if len(region) < 3:
        return None, None
    cells = [region]
    for cut in cuts:
        split = []
        for poly in cells:
            val = cut[0] + poly @ cut[1:]
            if min(val) < -5e-11 and max(val) > 5e-11:
                split.extend([clip(poly, cut), clip(poly, -cut)])
            else:
                split.append(poly)
        cells = split
    U, V = np.meshgrid(x, x, indexing="ij")
    W = (np.outer(w, w) * (1 - U)).ravel()
    U, V = U.ravel(), ((1 - U) * V).ravel()
    for poly in cells:
        center = np.r_[1.0, poly.mean(axis=0)]
        f = forms(center)
        g = f[1:]
        size = np.linalg.norm(g)
        if line:
            if not match(f):
                continue
            for j in range(2, len(poly)):
                a, b, c = poly[0], poly[j - 1], poly[j]
                v1, v2 = b - a, c - a
                jac = abs(v1[0] * v2[1] - v1[1] * v2[0])
                if jac > 1e-15:
                    samples.append(a + U[:, None] * v1 + V[:, None] * v2)
                    weights.append(jac * W)
        elif size > 2e-10:
            v = poly @ g + f[0] - target
            if min(v) > -1e-11 or max(v) < 1e-11:
                continue
            cross = []
            for j, p in enumerate(poly):
                if v[j] * v[j - 1] < 0:
                    cross.append(
                        poly[j - 1] + (p - poly[j - 1]) * (v[j - 1] / (v[j - 1] - v[j]))
                    )
                elif abs(v[j]) < 1e-11:
                    cross.append(p)
            if len(cross) < 2:
                continue
            a, b = max(
                combinations(cross, 2), key=lambda pts: np.linalg.norm(pts[0] - pts[1])
            )
            length = np.linalg.norm(b - a)
            if length > 1e-12:
                samples.append(a + x[:, None] * (b - a))
                weights.append(w * length / size)
    if not samples:
        return None, None
    nodes = np.concatenate(samples)
    return np.c_[np.ones(len(nodes)), nodes], np.concatenate(weights)


def spectral_law(q, activity, order=12):
    spec = q["spectrum"]

    def quadrature(bounds, cuts, dim, T, n, births, ends):
        one = np.eye(dim + 1)[0]

        def gate_form(point):
            f = np.zeros_like(one)
            for gate, sign in zip(spec["window"], (-1.0, 1.0)):
                for p, b in enumerate(births):
                    if gate <= point @ b:
                        continue
                    for c, v in enumerate(q["velocities"]):
                        end = ends[2 * p + c]
                        f += (
                            sign * v * ((end if point @ end < gate else gate * one) - b)
                        )
            return f

        return spectrum_nodes(bounds, cuts, dim, T, n, gate_form, spec)

    return packet_law(q, activity, order, node_rule=quadrature, gates=spec["window"])


def reading(q, activity, order=12, independent_read=False):
    table = tuple(slice(None) if c is None else c for c in q["multiplicities"])
    mass, moment = spectral_law(q, activity, order)
    selected = mass[table].sum()
    if q["observable"] == "waveform" and not independent_read:
        return moment[table].sum() / selected
    baseline, current = packet_law(q, activity, order)
    if q["observable"] == "spectrum":
        return selected / baseline[table].sum()
    return current[table].sum() / baseline[table].sum()
