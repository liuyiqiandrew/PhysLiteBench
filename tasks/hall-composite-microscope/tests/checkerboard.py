"""Periodic linear triangular elements on the actual 2×2 checkerboard cell."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def fields(materials, bias, gain, drive, n):
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x, y = x.ravel(), y.ravel()

    def index(dx, dy):
        return ((x + dx) % n) * n + (y + dy) % n

    ll, lr, ul, ur = index(0, 0), index(1, 0), index(0, 1), index(1, 1)
    p = ((x >= n // 2) ^ (y >= n // 2)).astype(int)
    J = []
    for m in materials:
        A = np.asarray(m["longitudinal"])
        H = gain * bias * np.asarray(m["hall"])
        J.append(np.block([[A, -H], [H, A]]))
    C = np.asarray(J)[p]
    E = np.asarray(drive).T.ravel()
    triangles = [
        ([ll, lr, ur], [[-1, 1, 0], [0, -1, 1]]),
        ([ll, ur, ul], [[0, 1, -1], [-1, 0, 1]]),
    ]
    elements = []
    rows, cols, vals = [], [], []
    rhs = np.zeros(2 * n * n)
    for nodes, gradient in triangles:
        B = np.kron(np.asarray(gradient) * n / 2, np.eye(2))
        dof = (2 * np.stack(nodes, axis=1)[:, :, None] + np.arange(2)).reshape(-1, 6)
        rows.append(np.repeat(dof, 6, axis=1).ravel())
        cols.append(np.tile(dof, (1, 6)).ravel())
        vals.append((B.T @ C @ B).ravel())
        np.add.at(rhs, dof, -(B.T @ C @ E))
        elements.append((dof, B))
    K = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(rhs), len(rhs)),
    ).tocsr()
    potential = np.zeros_like(rhs)
    potential[2:] = spsolve(K[2:, 2:], rhs[2:])
    gradients = np.concatenate([E + potential[dof] @ B.T for dof, B in elements])
    return gradients[:, :2] + 1j * gradients[:, 2:], np.tile(p, 2)


def probe(q, gain, n):
    fields1, labels = fields(q["materials"], q["bias"][0], gain, q["drives"][0], n)
    fields2, _ = fields(q["materials"], q["bias"][1], gain, q["drives"][1], n)
    a = fields1 @ q["contacts"][0]
    b = fields2 @ q["contacts"][1]
    mean = np.mean((labels == q["phase"]) * a.conj() * b)
    return float((np.exp(-1j * q["angle"]) * mean).real)
