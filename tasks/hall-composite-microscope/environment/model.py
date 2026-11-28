"""Exact terminal voltage statistics and deterministic mean load heat."""
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from film import signal
from thermal import terminal, voltage
from meter import Meter


class Model(Meter):
    def read(self, q):
        if q["mode"] == "noise":
            return voltage(q, self.gain)
        if q["mode"] != "heat":
            return signal(q, self.gain)
        A, _, start, equilibrium = terminal(q, self.gain)
        t = q["duration"]
        excess = start - equilibrium
        relax = expm(-A * t)
        integ = t * equilibrium + solve_continuous_lyapunov(
            A, excess - relax @ excess @ relax.T
        )
        L = np.asarray(q["load"])
        drift = q["baths"][1]["load"] * np.trace(np.linalg.solve(q["capacitance"], L))
        mean = np.trace(L @ integ) - drift * t
        return float(np.cos(q["tilt"] * mean - q["angle"]))
