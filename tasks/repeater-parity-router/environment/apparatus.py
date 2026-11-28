"""Calibrated source and a single outcome-resolved optical Bell swap."""
import math
import numpy as np

X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0])
I = np.eye(2)
PAULI = dict(X=X, Y=Y, Z=Z, I=I)


def rotate(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def pair(spec):
    beta, theta, phi = spec
    return np.diag([np.cos(beta), np.exp(1j * phi) * np.sin(beta)]) @ rotate(theta).T


def arm(outer, bank, angle, b, reverse=False, correction=True):
    """Corrected amplitude in (remote user, bank output) order, retaining the Born weight."""
    bell = np.array([[0.0, 1.0], [(-1.0) ** b, 0.0]]) / np.sqrt(2)
    corr = X @ (Z if b else I) if correction else I
    return (
        pair(outer)
        @ bell.conj()
        @ rotate(angle)
        @ (bank.T if reverse else bank)
        @ corr.T
    ).ravel()


class Calibrated:
    def fit(self, rows):
        exposures = np.array([r["input"]["exposure"] for r in rows])
        self.activity = float(
            -exposures @ np.log([r["value"] for r in rows]) / (exposures @ exposures)
        )
        return self

    def predict(self, inputs):
        return np.array(
            [
                math.exp(-self.activity * q["exposure"])
                if "exposure" in q
                else self.read(q)
                for q in inputs
            ],
            dtype=float,
        )
