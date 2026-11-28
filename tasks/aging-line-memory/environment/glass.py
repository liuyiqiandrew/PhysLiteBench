import numpy as np


class Readout:
    def fit(self, records):
        e = np.array([r["input"]["threshold"] for r in records])
        attenuation = -np.log([r["value"] for r in records])
        self.depth_scale = float(np.clip((e @ e) / (e @ attenuation), 0.9, 1.1))
        return self

    def predict(self, inputs):
        return np.asarray(
            [
                np.exp(-q["threshold"] / self.depth_scale)
                if "threshold" in q
                else self.memory(q)
                for q in inputs
            ],
            dtype=float,
        )


def one_temperature(q, tg):
    # Poisson-Dirichlet probabilities assign new or existing lines when the
    # temperature is common and the frozen landscape has no applied field.
    draws = q["preparations"]
    x = np.mean([q["settings"][i][0] for i in draws]) / tg
    paths = [([0], [1], 1.0)]
    for n in range(1, len(draws)):
        next_paths = []
        for seq, count, weight in paths:
            for j, m in enumerate(count):
                sizes = count.copy()
                sizes[j] += 1
                next_paths.append((seq + [j], sizes, weight * (m - x) / n))
            next_paths.append(
                (seq + [len(count)], count + [1], weight * x * len(count) / n)
            )
        paths = next_paths
    den = num = 0.0
    for seq, _, w in paths:
        if all((seq[a] == seq[b]) == same for a, b, same in q["gate"]):
            den += w
            if seq[-1] == seq[q["mark"]]:
                num += w
    return num / den
