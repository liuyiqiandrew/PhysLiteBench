"""The source qubit loses its off-diagonal coherence before the eraser."""
import numpy as np
from apparatus import Calibrated, arm, pair, PAULI


class Model(Calibrated):
    def read(self, q):
        if q["observable"] == "marker":
            return 0.5
        op = np.eye(1)
        for word in q["spin"]:
            for name in word:
                op = np.kron(op, PAULI[name])
        accepted = signal = 0.0
        for bank in map(pair, q["bank"]):
            for b in (0, 1):
                for d in (0, 1):
                    if q["report"] is not None and b ^ d != q["report"]:
                        continue
                    left = arm(q["outer"][0], bank, q["analyzers"][0], b)
                    right = arm(q["outer"][1], bank, q["analyzers"][1], d, reverse=True)
                    remote = np.kron(left, right)
                    accepted += np.vdot(remote, remote).real / 2
                    signal += np.vdot(remote, op @ remote).real / 2
        return accepted if q["observable"] == "herald" else signal / accepted
