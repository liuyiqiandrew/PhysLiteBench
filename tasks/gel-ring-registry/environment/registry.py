from functools import lru_cache
import numpy as np

REV = np.arange(4) ^ 1


def hits(p, z):
    v = np.array(p["routes"], float)
    q = v / v.sum() * (1 - 1 / (1 + z * p["feed"]))
    f = q.copy()
    for _ in range(160):
        excursion = q * f[REV]
        new = q / (1 - excursion.sum() + excursion)
        if max(abs(new - f)) < 1e-15:
            break
        f = new
    return f


@lru_cache(None)
def loops(n):
    words = np.arange(4, dtype=np.int8)[:, None]
    for i in range(1, n):
        codes = np.repeat(words, 3, axis=0)
        new = (np.tile(np.arange(1, 4), len(words)) + (codes[:, -1] ^ 1)) % 4
        words = np.column_stack((codes, new.astype(np.int8)))
    words = words[words[:, 0] != (words[:, -1] ^ 1)]
    unit = 4 ** np.arange(n - 1, -1, -1, dtype=np.int64)
    r = words @ unit
    key = r.copy()
    for j in range(1, n):
        r = (r % unit[0]) * 4 + r // unit[0]
        key = np.minimum(r, key)
    return words, np.unique(key, return_inverse=True)[1]


@lru_cache(None)
def pictures(n):
    w, c = loops(n)
    left = []
    right = []
    order = np.argsort(c, kind="stable")
    for group in np.split(order, np.flatnonzero(np.diff(c[order])) + 1):
        left.extend(np.repeat(group, len(group)))
        right.extend(np.tile(group, len(group)))
    return w, np.array(left), np.array(right)


def observable(q, z, pair_weights):
    w, ix, iy = pictures(q["length"])
    ok = np.isin(w[ix, 0], q["gate"])
    if q["shifted"]:
        ok &= ix != iy
    u = w[ix[ok]]
    v = w[iy[ok]]
    f0, f1, f2 = [hits(k, z) for k in [q["trunk"], *q["arms"]]]
    p = pair_weights(f0, f1, f2, u, v)
    if "probe" in q:
        obs = np.isin(v[:, -1], q["probe"])
    else:
        angles = np.array([q["flux"][0], -q["flux"][0], q["flux"][1], -q["flux"][1]])
        obs = 0.5 + 0.5 * np.cos(angles[v].sum(1) + q["phase"])
    return float(p @ obs / p.sum())


class Printer:
    def fit(self, records):
        dose = np.array([r["input"]["dose"] for r in records])
        odds = 1 / np.array([r["value"] for r in records]) - 1
        self.activity = float(np.clip((dose @ odds) / (dose @ dose), 0.75, 1.15))
        return self

    @staticmethod
    def pair_weights(f0, f1, f2, u, v):
        # Grow the daughter routes from the starting clasp, removing the contribution of the common lay.
        return f1[u].prod(axis=1) * f2[v].prod(axis=1)

    def predict(self, inputs):
        return np.array(
            [
                1 / (1 + self.activity * q["dose"])
                if "dose" in q
                else observable(q, self.activity, self.pair_weights)
                for q in inputs
            ],
            dtype=float,
        )
