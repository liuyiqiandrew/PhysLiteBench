"""Independent finite-Fock checks for the physical three-resonator Hamiltonian."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
A = np.array([[1.8, 0.21, -0.13], [0.21, 2.2, 0.17], [-0.13, 0.17, 2.6]])
B = np.array([[1.2, 0.1, 0.18], [0.1, 1.6, -0.14], [0.18, -0.14, 1.7]])

def fock(cutoff, s, temperatures, scale=1.04, levels=120):
    all_e = []
    all_n = []
    for parity in [0, 1]:
        basis = [(n0, n1, n2) for n0 in range(cutoff + 1) for n1 in range(cutoff + 1 - n0) for n2 in range(cutoff + 1 - n0 - n1) if (n0 + n1 + n2) % 2 == parity]
        ix = {n: i for i, n in enumerate(basis)}
        rows = []
        cols = []
        data = []

        def add(i, j, v):
            rows.append(i)
            cols.append(j)
            data.append(v)
        for col, n in enumerate(basis):
            add(col, col, sum((A[j, j] * n[j] for j in range(3))))
            for i in range(3):
                for j in range(3):
                    if i != j and n[j] > 0:
                        m = list(n)
                        m[i] += 1
                        m[j] -= 1
                        add(ix[tuple(m)], col, A[i, j] * np.sqrt((n[i] + 1) * n[j]))
                for j in range(i, 3):
                    m = list(n)
                    m[i] += 1
                    m[j] += 1
                    if sum(m) > cutoff:
                        continue
                    value = s * B[i, j] * np.sqrt((n[i] + 1) * (n[j] + 1)) if i != j else 0.5 * s * B[i, i] * np.sqrt((n[i] + 1) * (n[i] + 2))
                    row = ix[tuple(m)]
                    add(row, col, value)
                    add(col, row, value)
        matrix = coo_matrix((data, (rows, cols)), shape=(len(basis),) * 2).tocsr()
        en, vec = eigsh(matrix, k=min(levels, len(basis) - 2), which='SA', tol=2e-12)
        order = np.argsort(en)
        en = en[order]
        vec = vec[:, order]
        n = (np.array(basis).T @ (vec * vec)).T
        all_e.append(en)
        all_n.append(n)
    e = np.concatenate(all_e)
    n = np.concatenate(all_n)
    ix = np.argsort(e)
    e = e[ix]
    n = n[ix]
    values = []
    for t in temperatures:
        weights = np.exp(-scale * (e - e[0]) / t)
        values.append((weights @ n / weights.sum() - n[0]).tolist())
    return (values, e[-1] - e[0], len(e))
