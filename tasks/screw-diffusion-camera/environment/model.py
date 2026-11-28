import numpy as np
from scipy.linalg import expm
from apparatus import Camera
from passive import fringe


def spin(vector):
    x, y, z = vector
    return np.array([[z, x - 1j * y], [x + 1j * y, -z]])


class Model(Camera):
    def fringe(self, q):
        optical = fringe(q, self.mobility)
        if "replay" not in q:
            return optical
        # A gyro smooths all thermal kicks to their zero means. Keep the
        # programmed pulses, coherent control port and actual analyzer.
        s = q["replay"]
        identity = np.eye(2)
        P, Q = [expm(-0.5j * spin(s[k])) for k in ("forward_pulse", "replay_pulse")]
        A = (P + np.exp(1j * s["control_phase"]) * Q) / 2
        rho = (identity + spin(s["initial"])) / 2
        D = identity if s["analyzer"] is None else (identity + spin(s["analyzer"])) / 2
        return optical * np.trace(D @ A @ rho @ A.conj().T).real
