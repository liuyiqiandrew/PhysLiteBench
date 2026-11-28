"""Direct finite pore tree, with two chemical states at each actual reduced route."""
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve
from collections import Counter


def full(p, z, D, n):
    f = np.asarray(p["feed"])
    s = 1 / (1 + z * f)
    w = np.array(p["routes"])
    w = w / w.sum(1)[:, None]
    words = [()]
    ids = {(): 0}
    rows = []
    cols = []
    vals = []
    for i, a in enumerate(words):
        for j in range(4):
            b = a[:-1] if a and a[-1] == (j ^ 1) else a + (j,)
            if len(b) > D:
                continue
            if b not in ids:
                ids[b] = len(words)
                words.append(b)
            dest = ids[b]
            flip = p["switch"][j]
            for k in (0, 1):
                for h in (0, 1):
                    rows.append(2 * dest + h)
                    cols.append(2 * i + k)
                    vals.append(
                        (1 - s[k]) * w[k, j] * (flip[k] if h != k else 1 - flip[k])
                    )
    N = 2 * len(words)
    P = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
    x = np.zeros(N)
    x[:2] = [1 - p["blue"], p["blue"]]
    stop = spsolve(eye(N, format="csc") - P, x).reshape(-1, 2) @ s
    out = Counter()
    for a, v in zip(words, stop):
        while len(a) > 1 and a[0] == (a[-1] ^ 1):
            a = a[1:-1]
        if len(a) == n:
            key = min(a[i:] + a[:i] for i in range(n))
            out[key, a[0], a[-1]] += v
    return out


def check(q, z, D):
    A, B = [full(p, z, D, q["length"]) for p in q["rings"]]
    first = Counter()
    for (k, a, b), v in A.items():
        if a in q["gate"]:
            first[k] += v
    den = sum(first[k] * v for (k, a, b), v in B.items())
    num = sum(first[k] * v for (k, a, b), v in B.items() if b in q["probe"])
    return num / den
