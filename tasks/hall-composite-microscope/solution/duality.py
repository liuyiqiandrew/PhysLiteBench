"""Planar duality and the phase-resolved products of two local fields."""
import numpy as np


def square_root(a):
    val, vec = np.linalg.eigh(a)
    return (vec * np.sqrt(val)) @ vec.T


def experiments(q, scale, field, drive):
    A = [np.asarray(m["longitudinal"]) for m in q["materials"]]
    H = [scale * field * np.asarray(m["hall"]) for m in q["materials"]]
    root = square_root(A[0])
    inv = np.linalg.inv(root)
    z = inv @ (H[0] - H[1] + 1j * A[1]) @ inv
    ident = np.eye(2)
    w = (z - 1j * ident) @ np.linalg.inv(z + 1j * ident)
    w = (w + w.T) / 2
    # W times complex conjugation is real symmetric on the doubled space.
    block = np.block([[w.real, w.imag], [w.imag, -w.real]])
    vals, vec = np.linalg.eigh(block)
    chosen = vals > 1e-12
    s = vals[chosen]
    U = vec[:2, chosen] + 1j * vec[2:, chosen]
    if len(s) == 0:
        U = ident
    elif len(s) == 1:
        v = U[:, 0].conj()
        U = np.column_stack((U, [-v[1], v[0]]))
    s = np.pad(s, (0, 2 - len(s)))
    x, y = U.T.real, U.T.imag
    c = np.sqrt((1 + s) / (1 - s))

    def electric(k):
        return inv @ (x.T - 1j * y.T * k)

    lab = np.array(drive) @ np.array([1, 1j])
    e = np.linalg.solve(electric(c), lab)
    f = -H[0] @ lab + root @ (y.T + 1j * x.T * c) @ e
    return -1j * f, c, e, [electric(np.ones(2)), electric(c * c)]


def signal(q, scale, independent=False):
    if q["mode"] == "bulk":
        current = experiments(q, scale, q["bias"], q["drive"])[0]
        return float((np.exp(-1j * q["angle"]) * np.dot(q["contact"], current)).real)
    p = q["phase"]
    results = [experiments(q, scale, b, e) for b, e in zip(q["bias"], q["drives"])]
    (_, c, e, L), (_, d, g, M) = results
    left = np.asarray(q["contacts"][0]) @ L[p]
    right = np.asarray(q["contacts"][1]) @ M[p]
    if independent:
        a = left @ ((c if p == 0 else 1.0) * e / (1 + c))
        b = right @ ((d if p == 0 else 1.0) * g / (1 + d))
        total = 2 * a.conjugate() * b
    else:
        raw = np.outer(e.conj(), g)
        ratio = np.outer(c, d)
        part = raw.real / (c[:, None] + d[None, :]) + 1j * raw.imag / (1 + ratio)
        if p == 0:
            part *= ratio
        total = left.conj() @ part @ right
    return float((np.exp(-1j * q["angle"]) * total).real)
