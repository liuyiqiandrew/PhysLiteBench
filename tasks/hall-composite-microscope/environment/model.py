"""Exact film readings and equilibrium voltage fluctuations at the mean bath temperature."""
import numpy as np
from scipy.linalg import expm
from film import experiments, signal
from meter import Meter


class Model(Meter):
    def read(self, q):
        if q["mode"] != "noise":
            return signal(q, self.gain)
        b = q["bias"][1]
        currents = [experiments(q, self.gain, b, e.reshape(2, 2).T)[0] for e in np.eye(4)]
        bulk = np.column_stack([np.r_[j.real, j.imag] for j in currents])
        C = np.asarray(q["capacitance"])
        A = np.linalg.solve(C, bulk + np.array(q["load"]))
        last = q["baths"][1]
        mean = np.mean([*np.ravel(last["phase_modes"]), last["load"]])
        u, v = q["contacts"]
        return float(mean * np.asarray(u) @ np.linalg.solve(C, expm(-q["lag"] * A).T) @ v)
