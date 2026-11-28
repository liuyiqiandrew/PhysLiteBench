from functools import lru_cache
import numpy as np


@lru_cache(None)
def necklaces(n):
    out = {}
    letters = [0] * (n + 1)

    def rec(t, period):
        if t > n:
            if n % period == 0 and letters[1] != (letters[n] ^ 1):
                out.setdefault(period, []).append(letters[1:].copy())
            return
        base = letters[t - period]
        for k in range(base, 4):
            if t > 1 and (letters[t - 1] ^ 1) == k:
                continue
            letters[t] = k
            rec(t + 1, period if k == base else t)

    rec(1, 1)
    return {d: np.array(out[d], dtype=np.int8) for d in out}


def green(p, z):
    f = np.array(p["feed"])
    end = 1 / (1 + z * f)
    route = np.array(p["routes"], dtype=float)
    route /= route.sum(1)[:, None]
    trans = []
    for i, sw in enumerate(p["switch"]):
        mat = np.array([[1 - sw[0], sw[0]], [sw[1], 1 - sw[1]]])
        trans.append(np.diag((1 - end) * route[:, i]) @ mat)
    eye = np.eye(2)
    G = [eye] * 4
    for size in range(85):
        loops = [trans[k] @ G[k ^ 1] @ trans[k ^ 1] for k in range(4)]
        G = [
            np.linalg.inv(eye - sum(l for k, l in enumerate(loops) if k != j))
            for j in range(4)
        ]
    F = [G[i] @ trans[i] for i in range(4)]
    shut = np.linalg.solve(eye - sum(loops), end)
    root = np.outer(shut, [1 - p["blue"], p["blue"]])
    total = np.zeros((4, 2, 2))
    level = [F[k ^ 1] @ root @ F[k] for k in range(4)]
    for size in range(80):
        total += level
        level = [
            F[i ^ 1] @ sum(v for j, v in enumerate(level) if (j ^ 1) != i) @ F[i]
            for i in range(4)
        ]
    tails = np.array(
        [
            [
                root + sum(A for k, A in enumerate(total) if k != l and k != (h ^ 1))
                for l in range(4)
            ]
            for h in range(4)
        ]
    )
    return np.array(F), tails


def reference(q, z):
    ring = [green(p, z) for p in q["rings"]]
    num = den = 0.0
    for d, words in necklaces(q["length"]).items():
        vals = [np.zeros(len(words)) for _ in range(3)]
        for k in range(d):
            w = np.roll(words, k, axis=1)
            for j, (F, tail) in enumerate(ring):
                work = tail[w[:, 0], w[:, -1]]
                for t in range(w.shape[1] - 1, -1, -1):
                    work = F[w[:, t]] @ work
                prob = np.trace(work, axis1=1, axis2=2)
                if j == 0:
                    vals[0] += prob * np.isin(w[:, 0], q["gate"])
                else:
                    vals[1] += prob
                    if "probe" in q:
                        vals[2] += prob * np.isin(w[:, -1], q["probe"])
                    else:
                        direction = np.array(
                            [q["flux"][0], -q["flux"][0], q["flux"][1], -q["flux"][1]]
                        )
                        vals[2] += prob * (
                            0.5 + 0.5 * np.cos(direction[w].sum(1) + q["phase"])
                        )
        num += vals[0] @ vals[2]
        den += vals[0] @ vals[1]
    return num / den
