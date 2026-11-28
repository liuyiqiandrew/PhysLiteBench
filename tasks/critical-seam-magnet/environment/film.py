import numpy as np


def device(q, gain, method, separate=False):
    pol = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    energy = pol @ q["bias"] + q["coupling"] * pol.prod(axis=1)
    w = np.exp(gain * energy)
    Za = []
    means = []
    for f in q["films"]:
        Q, Z = method(f)
        # Each marked vector and partition vector already has its welded component normalized to one.
        Za.append(Z)
        means.append(Q / Z)
    if separate:
        distributions = [w * z / np.sum(w * z) for z in Za]
        am, bm = [v @ m for v, m in zip(distributions, means)]
        if q["read"] == "overlap":
            return float(am * bm)
        if q["read"] == "connected":
            return 0.0
        direction = pol[:, q["dial"]]
        d = []
        for v, m in zip(distributions, means):
            d.append(gain * (v @ (m * direction) - (v @ m) * (v @ direction)))
        return float(d[0] * bm + am * d[1])
    v = w * Za[0] * Za[1]
    v /= v.sum()
    a, b = means
    pair = a * b
    if q["read"] == "overlap":
        return float(v @ pair)
    if q["read"] == "connected":
        return float(v @ pair - (v @ a) * (v @ b))
    direction = pol[:, q["dial"]]
    return float(gain * (v @ (pair * direction) - (v @ pair) * (v @ direction)))
