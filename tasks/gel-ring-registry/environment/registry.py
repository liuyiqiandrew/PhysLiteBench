from functools import lru_cache
import numpy as np


def transitions(p, activity):
    weights = np.array(p["routes"], float)
    weights /= weights.sum(axis=1)[:, None]
    stop = 1 / (1 + activity * np.array(p["feed"]))
    Q = np.zeros((4, 2, 2))
    for j in range(4):
        f, g = p["switch"][j]
        Q[j] = ((1 - stop) * weights[:, j])[:, None] * [[1 - f, f], [g, 1 - g]]
    return Q, stop, np.array([1 - p["blue"], p["blue"]])


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


def ring_weights(p, z, words, dynamics):
    F, S = dynamics(p, z)
    mat = F[words[:, 0]]
    for j in range(1, words.shape[1]):
        mat = mat @ F[words[:, j]]
    return np.einsum("nij,nji->n", mat, S[words[:, 0], words[:, -1]])


def shot(q, z, dynamics):
    words, classes = loops(q["length"])
    w, v = [ring_weights(p, z, words, dynamics) for p in q["rings"]]
    g = np.isin(words[:, 0], q["gate"])
    w = w * g
    group = classes
    left = np.bincount(group, weights=w)[group]
    coupled = left * v
    den = coupled.sum()
    if "probe" in q:
        obs = np.isin(words[:, -1], q["probe"])
    else:
        phase = q["flux"][0] * ((words == 0).sum(1) - (words == 1).sum(1))
        phase += q["flux"][1] * ((words == 2).sum(1) - (words == 3).sum(1))
        obs = 0.5 * (1 + np.cos(q["phase"] + phase))
    return float(coupled @ obs / den), float(den / (max(w.sum(), 1e-300) * v.sum()))


class Printer:
    def fit(self, records):
        dose = np.array([r["input"]["dose"] for r in records])
        odds = 1 / np.array([r["value"] for r in records]) - 1
        self.activity = float(np.clip((dose @ odds) / (dose @ dose), 0.75, 1.15))
        return self

    @staticmethod
    def dynamics(p, z):
        # Count exactly the exposed core; its raw contour has no erased excursions.
        Q, stop, pi = transitions(p, z)
        return Q, np.broadcast_to(np.outer(stop, pi), (4, 4, 2, 2))

    def predict(self, inputs):
        return np.array(
            [
                1 / (1 + self.activity * q["dose"])
                if "dose" in q
                else shot(q, self.activity, self.dynamics)[0]
                for q in inputs
            ],
            dtype=float,
        )
