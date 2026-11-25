import numpy as np


def predict_at(experiments, rate_scale):
    output = []
    for experiment in experiments:
        energies = np.asarray(experiment['energies'], dtype=float)
        attempts = np.asarray(experiment['attempts'], dtype=float)
        affinities = np.asarray(experiment['affinities'], dtype=float)
        difference = np.roll(energies, -1) - energies
        forward = rate_scale * attempts * np.exp((affinities[None, :] - difference[:, None]) / 2)
        reverse = rate_scale * attempts * np.exp(-(affinities[None, :] - difference[:, None]) / 2)
        f, r = forward.sum(axis=1), reverse.sum(axis=1)
        generator = np.zeros((3, 3))
        for i in range(3):
            j = (i + 1) % 3
            generator[j, i] += f[i]
            generator[i, j] += r[i]
            generator[i, i] -= f[i]
            generator[j, j] -= r[i]
        system = generator.copy()
        system[-1] = 1
        probability = np.linalg.solve(system, [0., 0., 1.])
        positive = probability[:, None] * forward
        negative = np.roll(probability, -1)[:, None] * reverse
        a, b = positive.sum(axis=1), negative.sum(axis=1)
        output.append(float(np.sum((a - b) * np.log(a / b))))
    return np.asarray(output)


class Model:
    def __init__(self):
        self.rate_scale = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        uncertainty = np.asarray([r['sigma'] for r in records])
        response = predict_at(experiments, 1.) / uncertainty
        measured = np.asarray([r['value'] for r in records]) / uncertainty
        self.rate_scale = float(np.clip(response @ measured / (response @ response), .6, 1.4))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.rate_scale)
