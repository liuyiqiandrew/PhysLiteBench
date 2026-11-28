import numpy as np
from registry import REV


def joint(f0, f1, f2, u, v):
    n = u.shape[1]
    ids = np.arange(len(u))
    k = (u == v).cumprod(axis=1).sum(axis=1)
    side = f0 * f1[REV] * f2[REV]
    T = np.eye(4) - np.tile(side, (4, 1))
    T[np.arange(4), REV] += side[REV]
    branches = side * np.linalg.solve(T, np.ones(4))

    def prod(f, w):
        return np.column_stack((np.ones(len(w)), np.cumprod(f[w], axis=1)))

    def margin(w, other, first, second, shared):
        trunk = prod(f0, w)
        dst = prod(first, w)
        sibling = prod(second, other)
        rev = prod(second[REV], w)
        weight = np.zeros(len(u))
        for j in range(n + 1):
            cond = (j <= k) if shared else (j > k)
            if not cond.any():
                continue
            active = np.ones((len(u), 4), bool)
            if j:
                active[ids, REV[w[:, j - 1]]] = False
            if j < n:
                active[ids, w[:, j]] = False
                if shared:
                    active[ids, other[:, j]] = False
            endpoint = trunk[:, j] * dst[:, -1] / dst[:, j]
            if shared:
                endpoint *= sibling[:, -1] / sibling[:, j]
            else:
                endpoint *= rev[:, j] / rev[ids, k] * sibling[:, -1] / sibling[ids, k]
            weight += cond * endpoint * (1 + active @ branches)
        return weight

    return (
        margin(u, v, f1, f2, True)
        + margin(u, v, f1, f2, False)
        + margin(v, u, f2, f1, False)
    )
