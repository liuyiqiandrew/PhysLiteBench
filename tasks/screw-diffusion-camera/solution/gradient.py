"""Weak-field pickup for a closed gyroscope polygon in a linear field gradient."""
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply
from numpy.polynomial.hermite import hermgauss
from apparatus import ROT, SCREWS, TRANS
from passive import operators, project


def pickup(q, nu, order=7, nodes=18, body_drive=True):
    s = q["pickup"]
    a, b = s["axes"]
    d = s["end"]
    mid = s["middle"]
    power = 1 if s["read"] == "slope" else 2

    def expectation(waves, order, r):
        pos, angular = operators(order)
        n = (order + 1) ** 2
        space = eye(n, format="csr")
        gates = []
        for v in q["gates"]:
            g = eye(pos[0].shape[0], format="csr") + project(pos, v)
            gates.append((g @ g)[:n, :n] / 4)
        # x and m are the accumulated horizontal coordinate and 1/2 ∫x²dy.
        basis = [(i, j) for j in range(r + 1) for i in range(2 * (r - j) + 1)]
        size = len(basis)
        poly = eye(size, format="csr")
        moment_derivative = np.zeros((size, size))
        Dx = np.zeros((size, size))
        B = Dx.copy()
        for k, (i, j) in enumerate(basis):
            if i:
                Dx[basis.index((i - 1, j)), k] = i
            if j:
                B[basis.index((i + 2, j - 1)), k] = j / 2
                moment_derivative[basis.index((i, j - 1)), k] = j
        Dx = csr_matrix(Dx)
        B = csr_matrix(B)

        def operators_for(k):
            L = -k * k * sum(rate * (p @ p) for p, rate in zip(pos, TRANS))
            twist = []
            for i in range(3):
                v = q["hand"] * SCREWS[i] + np.cross(np.eye(3)[i], q["tag"])
                A = angular[i] + 1j * k * project(pos, v)
                L += ROT[i] * (A @ A)
                twist.append(A[:n, :n])
            # A [A,B] A^-1 [A,B]^-1 is the third-order Lie bracket.
            # For body screws a Lie bracket acts on both the angular and
            # linear parts of their twists.
            omega = s["stroke"]
            if body_drive:
                ea, eb = np.eye(3)[a], np.eye(3)[b]
                ha, hb = q["hand"] * SCREWS[[a, b]]
                wab = np.cross(ea, eb)
                hab = np.cross(ea, hb) - np.cross(eb, ha)
                w = np.cross(ea, wab)
                v = np.cross(ea, hab) - np.cross(wab, ha)
                twist_drive = project(angular, w) + 1j * k * project(pos, v + np.cross(w, q["tag"]))
                L += (omega / nu) * ROT[a] * np.sqrt(ROT[b]) * twist_drive
            G = kron(L[:n, :n], poly, format="csr")
            G += (omega / nu) * kron(space, csr_matrix(moment_derivative), format="csr")
            G += kron(space, Dx @ Dx + B @ B, format="csr")
            G += 2 * np.sqrt(ROT[a]) * kron(twist[a], Dx, format="csr") + 2 * np.sqrt(ROT[b]) * kron(
                twist[b], B, format="csr"
            )
            coupled = 2j * (
                np.sqrt(ROT[b]) * kron(twist[b], poly, format="csr") + kron(space, B, format="csr")
            )
            return G, coupled

        K1, K2 = map(operators_for, waves)
        t1, t2 = q["times"]
        gate1 = kron(gates[1], poly, format="csr")
        tail = np.zeros((n, size), complex)
        from math import comb

        for j in range(r + 1):
            tail[0, basis.index((2 * j, r - j))] = comb(r, j) * (-d / 6) ** j
        tail = tail.ravel()

        def travel(v, K, t, p):
            return expm_multiply((nu * t) * (K[0] + p * K[1]), v)

        x, w = hermgauss(nodes)

        def pinned(fun, t, displacement):
            scale = np.sqrt(nu * t)
            accum = 0j
            for z, wt in zip(x, w):
                p = z / scale
                accum += wt * np.exp(-1j * p * displacement) * fun(p)
            return np.exp(displacement * displacement / (4 * nu * t)) * accum / np.sqrt(np.pi)

        if mid is None:
            evolved = pinned(lambda p: travel(gate1 @ travel(tail, K2, t2, p), K1, t1, p), t1 + t2, d)
        else:
            rest = pinned(lambda p: travel(tail, K2, t2, p), t2, d - mid)
            weight = gate1 @ rest
            evolved = pinned(lambda p: travel(weight, K1, t1, p), t1, mid)
        return (gates[0].getrow(0) @ evolved.reshape(n, size))[0, 0]

    raw = expectation(q["wave"], order, power)
    norm = expectation([0.0, 0.0], 2, 0)
    return float(((1j if power == 1 else 1.0) * np.exp(-1j * q["phase"]) * raw / norm).real) / (
        nu * sum(q["times"])
    ) ** (1.5 * power)
