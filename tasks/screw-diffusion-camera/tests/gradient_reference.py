"""Exact finite screw, gyroscope ordinate and polygon-moment increments."""
import numpy as np
from math import comb
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply
from reference import grid, rates, screws, harmonics
from scipy.spatial.transform import Rotation


def reference_gradient(q, nu, order=7, h=0.002, nodes=35):
    s = q["pickup"]
    a, b = s["axes"]
    d = s["end"]
    mid = s["middle"]
    r = 1 if s["read"] == "slope" else 2

    def raw(waves, order, r):
        u, w, Y, adj, neigh = grid(order, h)
        tag = u @ q["tag"]
        N = Y.shape[1]
        k1, k2 = waves
        t1, t2 = q["times"]
        g0, g1 = [(1 + u @ v) ** 2 / 4 for v in q["gates"]]
        left = (w * g0 * np.exp(-1j * k1 * tag)) @ Y
        midgate = csr_matrix(adj @ ((g1 * np.exp(1j * (k1 - k2) * tag))[:, None] * Y))
        basis = sorted([(i, j) for i in range(2 * r + 1) for j in range(r + 1) if i + 2 * j <= 2 * r])
        M = len(basis)
        Is = eye(M, format="csr")

        def move(ax, amount):
            delta = amount / np.sqrt(rates[ax])
            out = np.zeros((M, M))
            for col, (i, j) in enumerate(basis):
                if ax == a:
                    for l in range(i + 1):
                        out[basis.index((l, j)), col] = comb(i, l) * delta ** (i - l)
                elif ax == b:
                    for l in range(j + 1):
                        out[basis.index((i + 2 * l, j - l)), col] = comb(j, l) * (delta / 2) ** l
                else:
                    out[col, col] = 1.0
            return csr_matrix(out)

        def terms(k):
            trans = csr_matrix(adj @ ((-k * k * (u**2 @ [0.18, 0.27, 0.36]))[:, None] * Y))
            G = kron(trans, Is, format="csr") - 2 * sum(rates) / h**2 * eye(M * N, format="csr")
            for j, ns in enumerate(neigh):
                v = []
                for (mat, disp), angle in zip(ns, (h, -h)):
                    orb = csr_matrix(adj @ (np.exp(1j * k * q["hand"] * disp)[:, None] * mat))
                    v.append((rates[j] / h**2) * kron(orb, move(j, angle), format="csr"))
                G += v[0] + v[1]
                if j == b:
                    even = v[0] + v[1]
                    odd = v[0] - v[1]
            # Apply the ten actual stroke segments. The central difference
            # extracts their order-three net rigid motion and signed moment.
            edge = 0.004
            drives = []
            loop = [(a, 1), (b, 1), (a, -1), (b, -1)]
            word = [(a, 1)] + loop + [(a, -1)] + [(j, -t) for j, t in reversed(loop)]
            for side in (edge, -edge):
                rot = np.eye(3)
                disp = np.zeros(3)
                x = y = m = 0.0
                for j, sign in word:
                    length = side * sign
                    axis = np.eye(3)[j]
                    angle = length * np.sqrt(rates[j])
                    translation = q["hand"] * screws[j]
                    parallel = axis * (axis @ translation)
                    local = (
                        angle * parallel
                        + np.sin(angle) * (translation - parallel)
                        + (1 - np.cos(angle)) * np.cross(axis, translation)
                    )
                    disp += rot @ local
                    rot = rot @ Rotation.from_rotvec(axis * angle).as_matrix()
                    if j == a:
                        x += length
                    else:
                        m += x * x * length / 2
                        y += length
                at = csr_matrix(adj @ (np.exp(1j * k * (u @ disp))[:, None] * harmonics(u @ rot, order)))
                update = np.zeros((M, M))
                for col, (i, j) in enumerate(basis):
                    for l in range(j + 1):
                        update[basis.index((i, l)), col] = comb(j, l) * m ** (j - l)
                drives.append(kron(at, csr_matrix(update), format="csr"))
            G += s["stroke"] * (drives[0] - drives[1]) / (nu * 2 * edge**3)
            for x in [G, even, odd]:
                x.data[abs(x.data) < 8.0e-8] = 0
                x.eliminate_zeros()
            return G, even, odd

        G1, G2 = map(terms, waves)
        ich = eye(M * N, format="csr")
        gate = kron(midgate, Is, format="csr")
        poly = np.zeros(M)
        for l in range(r + 1):
            poly[basis.index((2 * l, r - l))] = comb(r, l) * (-d / 6) ** l
        tail = np.kron(adj @ np.exp(1j * k2 * tag), poly)

        def evolve(v, T, t, p):
            beta = h * p / np.sqrt(rates[b])
            G = T[0] - 2 * np.sin(beta / 2) ** 2 * T[1] + 1j * np.sin(beta) * T[2] + p * p * ich
            return expm_multiply(t * nu * G, v)

        zs = np.linspace(-7.2, 7.2, nodes)

        def integral(f, t, y):
            scale = np.sqrt(nu * t)
            out = 0j
            for z in zs:
                p = z / scale
                out += np.exp(-z * z - 1j * p * y) * f(p)
            return out * (zs[1] - zs[0]) * np.exp(y * y / (4 * nu * t)) / np.sqrt(np.pi)

        if mid is None:
            v = integral(lambda p: evolve(gate @ evolve(tail, G2, t2, p), G1, t1, p), t1 + t2, d)
        else:
            v = gate @ integral(lambda p: evolve(tail, G2, t2, p), t2, d - mid)
            v = integral(lambda p: evolve(v, G1, t1, p), t1, mid)
        return left @ v.reshape(N, M)[:, basis.index((0, 0))]

    value = raw(q["wave"], order, r)
    den = raw([0.0, 0.0], 2, 0)
    return float(((1j if r == 1 else 1) * np.exp(-1j * q["phase"]) * value / den).real) / (
        nu * sum(q["times"])
    ) ** (1.5 * r)
