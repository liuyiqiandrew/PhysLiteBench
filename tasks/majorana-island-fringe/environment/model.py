import numpy as np

from apparatus import Readout, contacts, polarizations


class Model(Readout):
    def fringe(self, q):
        # Static mean-field approximation to the island during the pulses.
        t = polarizations(q)
        other = np.array([np.prod(np.delete(t, j)) for j in range(len(t))])
        mean = (t + q["parity"] * other) / (1 + q["parity"] * np.prod(t))
        u = contacts(q)
        amplitudes = []
        for arm in q["arms"]:
            accumulated = np.zeros(2)
            for a, b, turn, coupling in arm:
                charge = (u[a, ::2] * u[b, 1::2] - u[a, 1::2] * u[b, ::2]) @ mean
                accumulated += charge * np.array([turn + coupling, turn - coupling])
            branch = np.exp(-.5j * accumulated)
            amplitudes.append(branch[0] + q["outcome"] * np.exp(-1j * q["meter_phase"]) * branch[1])
        p = sum(abs(z)**2 for z in amplitudes)
        if p < 1e-20:
            return 0.
        cross = amplitudes[0].conjugate() * amplitudes[1]
        return float(2 * (np.exp(-1j * q["phase"]) * cross).real / p)
