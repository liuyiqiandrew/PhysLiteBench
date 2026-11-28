"""Local rank of the two induced-charge readings on each affine history cell."""
import numpy as np
from itertools import combinations
from transport import packet_law, clip, rule


def cells(bounds, cuts, dim, T):
    if dim == 1:
        lo, hi = 0.0, T
        for a, b in bounds:
            if abs(b) < 1e-12:
                if a < -1e-12:
                    return []
            elif b > 0:
                lo = max(lo, -a / b)
            else:
                hi = min(hi, -a / b)
        if hi <= lo + 1e-12:
            return []
        marks = sorted(
            set(
                [lo, hi]
                + [-a / b for a, b in cuts if abs(b) > 1e-12 and lo < -a / b < hi]
            )
        )
        return [np.array([[a], [b]]) for a, b in zip(marks, marks[1:])]
    poly = np.array([[0.0, 0.0], [T, 0.0], [T, T], [0.0, T]])
    for bound in bounds:
        poly = clip(poly, bound)
    if len(poly) < 3:
        return []
    boxes = [poly]
    for cut in cuts:
        remaining = []
        for box in boxes:
            z = box @ cut[1:] + cut[0]
            if z.min() < -3e-11 and z.max() > 3e-11:
                remaining.extend([clip(box, cut), clip(box, -cut)])
            else:
                remaining.append(box)
        boxes = remaining
    return boxes


def quadrature(q, mode, z):
    desired = {"point": 0, "ridge": 1, "area": 2}[mode]

    def nodes(bounds, cuts, dim, T, order, births, ends):
        if dim < desired:
            return None, None
        x, w = rule(order)
        unit = np.eye(dim + 1)[0]
        samples = []
        weights = []
        for poly in cells(bounds, cuts, dim, T):
            p = np.r_[1.0, poly.mean(0)]
            form = np.zeros((2, dim + 1))
            for j, gate in enumerate(q["gates"]):
                for t, sign in zip(gate, [-1.0, 1.0]):
                    for k, b in enumerate(births):
                        if t <= p @ b:
                            continue
                        for c, v in enumerate(q["velocities"]):
                            e = ends[2 * k + c]
                            form[j] += sign * v * ((e if p @ e < t else t * unit) - b)
            A = form[:, 1:]
            offset = z - form[:, 0]
            L, S, R = np.linalg.svd(A, full_matrices=False)
            rank = np.sum(S > 2e-10)
            if rank != desired:
                continue
            if np.linalg.norm(offset - L[:, :rank] @ (L[:, :rank].T @ offset)) > 2e-9:
                continue
            if rank == 0:
                if dim == 1:
                    l, r = poly[:, 0]
                    samples.append((l + (r - l) * x)[:, None])
                    weights.append((r - l) * w)
                else:
                    X, Y = np.meshgrid(x, x, indexing="ij")
                    U = X.ravel()
                    V = ((1 - X) * Y).ravel()
                    W = (np.outer(w, w) * (1 - X)).ravel()
                    for k in range(2, len(poly)):
                        a, b, c = poly[0], poly[k - 1], poly[k]
                        u = b - a
                        v = c - a
                        jac = abs(u[0] * v[1] - u[1] * v[0])
                        if jac > 1e-14:
                            samples.append(a + u * U[:, None] + v * V[:, None])
                            weights.append(W * jac)
            elif rank == 2:
                point = np.linalg.solve(A, offset)
                delta = point - poly
                edge = np.roll(poly, -1, axis=0) - poly
                nonzero = np.linalg.norm(edge, axis=1) > 1.0e-10
                if (edge[:, 0] * delta[:, 1] - edge[:, 1] * delta[:, 0])[
                    nonzero
                ].min() > 1.0e-11:
                    samples.append(point[None, :])
                    weights.append(np.array([1 / np.prod(S)]))
            else:
                val = (poly @ A.T - offset) @ L[:, 0]
                if min(val) > -1e-11 or max(val) < 1e-11:
                    continue
                if dim == 1:
                    a, b = poly[:, 0]
                    point = a + (b - a) * val[0] / (val[0] - val[1])
                    samples.append(np.array([[point]]))
                    weights.append(np.array([1 / S[0]]))
                else:
                    crossing = []
                    for j, b in enumerate(poly):
                        a = poly[j - 1]
                        v1, v2 = val[j - 1], val[j]
                        if v1 * v2 < 0:
                            crossing.append(a + (b - a) * v1 / (v1 - v2))
                        elif abs(v2) < 1.0e-12:
                            crossing.append(b)
                    if len(crossing) < 2:
                        continue
                    a, b = max(
                        combinations(crossing, 2),
                        key=lambda pts: np.linalg.norm(pts[0] - pts[1]),
                    )
                    length = np.linalg.norm(b - a)
                    if length > 1e-12:
                        samples.append(a + (b - a) * x[:, None])
                        weights.append(w * length / S[0])
        if not samples:
            return None, None
        pts = np.concatenate(samples)
        return np.c_[np.ones(len(pts)), pts], np.concatenate(weights)

    return nodes


def values(q, a=0.92, order=9):
    m, w = packet_law(
        q,
        a,
        order,
        node_rule=quadrature(q, q["part"], np.array(q["charges"])),
        gates=sum(q["gates"], []),
    )
    idx = tuple(slice(None) if s is None else s for s in q["multiplicities"])
    p = m[idx].sum()
    base, current = packet_law(q, a, order)
    b = base[idx].sum()
    return (p / b, w[idx].sum() / p if p else None, current[idx].sum() / b)
