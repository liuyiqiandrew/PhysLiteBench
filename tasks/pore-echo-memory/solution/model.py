import numpy as np
from scipy.optimize import brentq


def single(private, common, diffusion):
    width = (2 * np.pi**2 * diffusion / (private + common))**(1/3)
    return 1.5 * (private + common) * width


def pair(row, common):
    private = np.array([row["a"], row["b"]])
    confinement = np.pi**2 * np.array([row["D_A"], row["D_B"]])
    spacing = row["s"]
    best = (2 * confinement / private)**(1/3)

    def slope(total):
        return common + np.minimum(private - 2 * confinement / total**3, 0).sum()

    total = max(spacing, brentq(slope, 1e-4, max(best)))
    widths = np.minimum(total, best)
    if widths.sum() < spacing:
        def split_slope(x):
            return (private[0] - private[1] - 2 * confinement[0] / x**3
                    + 2 * confinement[1] / (spacing - x)**3)
        left = brentq(split_slope, 1e-5, spacing - 1e-5)
        widths = np.array([left, spacing - left])
    return common * total + np.sum(private * widths + confinement / widths**2)


class Model:
    def fit(self, records):
        estimates = []
        for r in records:
            x = r["input"]
            rate = r["value"]
            estimates.append(np.sqrt((rate / 1.5)**3 / (2 * np.pi**2 * x["D_A"])) - x["a"])
        self.intensity = np.mean(estimates)
        return self

    def predict(self, inputs):
        return np.array([single(x["a"], self.intensity, x["D_A"]) if x["mode"] == "single"
                         else pair(x, self.intensity) for x in inputs])
