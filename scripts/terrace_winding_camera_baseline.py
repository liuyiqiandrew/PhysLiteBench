"""Starter approximation: match the lifted pins with independent random-walk bridges."""
import numpy as np
from scipy.special import ive


def image(q, rate):
    lower, upper = q["bottom"], q["top"]
    n, length, winding = len(lower), q["circumference"], q["winding"]
    first, second = np.array(q["levels"]) * q["height"]
    position = np.arange(-40, length + 41)
    site = position % length
    in_first = (q["window"][0] <= site) & (site < q["window"][1])
    in_second = (q["count_window"][0] <= site) & (site < q["count_window"][1])
    middle = ive(abs(position[:, None] - position), 2 * rate * (second - first))
    counts = np.array([1.])
    for label, start in enumerate(lower):
        lift, index = divmod(label + winding, n)
        end = upper[index] + length * lift
        left = ive(abs(position - start), 2 * rate * first)
        if label == q["tag"]:
            left *= in_first
        right = ive(abs(position - end), 2 * rate * (q["height"] - second))
        mass = (left @ middle) * right / ive(abs(end - start), 2 * rate * q["height"])
        counts = np.convolve(counts, [mass[~in_second].sum(), mass[in_second].sum()])
    return float(counts[q["count"]])


class Model:
    def fit(self, records):
        self.rate = float(np.mean([row["value"] / (2 * row["input"]["height"])
                                   for row in records]))
        return self

    def predict(self, inputs):
        return np.array([2 * self.rate * q["height"] if q["mode"] == "wandering"
                         else image(q, self.rate) for q in inputs], dtype=float)
