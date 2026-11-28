from registry import transitions
import numpy as np


def tightened(p, activity, reset=False):
    Q, stop, pi = transitions(p, activity)
    F = Q.copy()
    eye = np.eye(2)
    rev = np.arange(4) ^ 1
    for _ in range(180):
        back = Q @ F[rev]
        if reset:
            back = np.array([np.diag(x.sum(axis=1)) for x in back])
        common = eye - back.sum(axis=0)
        nxt = np.linalg.solve(common[None, :, :] + back, Q)
        if np.max(abs(nxt - F)) < 2e-15:
            break
        F = nxt
    d = np.linalg.solve(common, stop)
    root = np.outer(d, pi)
    ends = np.broadcast_to(root, (4, 2, 2))
    for _ in range(180):
        layers = F[rev] @ ends @ F
        nxt = root + layers.sum(axis=0) - layers[rev]
        if np.max(abs(nxt - ends)) < 2e-15:
            break
        ends = nxt
    layers = F[rev] @ ends @ F
    seam = np.zeros((4, 4, 2, 2))
    for a in range(4):
        for b in range(4):
            seam[a, b] = root + sum(
                (layers[k] for k in range(4) if k != (a ^ 1) and k != b),
                start=np.zeros((2, 2)),
            )
    return F, seam
