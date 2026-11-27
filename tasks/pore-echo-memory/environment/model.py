"""Four-trap echo model."""
import numpy as np


def step(p):
    walk = np.eye(4)
    return (1 - p) * walk + p / 2 * (np.roll(walk, 1, axis=1) + np.roll(walk, -1, axis=1))


def phase_moments(scan):
    signs = np.array(scan["signs"])
    hops = [step(p) for p in scan["hops"]]
    position = np.eye(4)[scan["start"]]
    mean_counts = np.zeros(4)
    variance = float(len(signs))
    for i, sign in enumerate(signs):
        mean_counts += sign * position
        propagator = np.eye(4)
        for j in range(i + 1, len(signs)):
            propagator = propagator @ hops[j - 1]
            match = position @ np.diag(propagator)
            variance += 2 * sign * signs[j] * match
        if i < len(hops):
            position = position @ hops[i]
    return variance, mean_counts


class Model:
    def __init__(self):
        self.width = None

    def fit(self, records):
        raise NotImplementedError("Fit width from the curvature data.")

    def predict_calibration(self, records):
        return np.array([self.width**2 * phase_moments(r["scan"])[0] for r in records])

    def predict(self, experiments):
        answers = []
        for row in experiments:
            a = (self.width * row["gain"])**2 / 2
            var, counts = phase_moments(row["scan"])
            if row["mode"] == "echo":
                answers.append(np.exp(-a * var))
            else:
                var2, counts2 = phase_moments(row["partner"])
                mixed = np.exp(-a * (var + var2 - 2 * counts @ counts2))
                answers.append(mixed - np.exp(-a * (var + var2)))
        return np.array(answers)
