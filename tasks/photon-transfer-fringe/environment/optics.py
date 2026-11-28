"""Passive mesh convention and calibration on a thermal input mixed with vacuum."""
import numpy as np


def mesh(size, elements):
    matrix = np.eye(size, dtype=complex)
    for i, j, angle, phase in elements:
        c = np.cos(angle)
        s = np.sin(angle) * np.exp(1j * phase)
        matrix[[i, j]] = np.array([[c, -s.conjugate()], [s, c]]) @ matrix[[i, j]]
    return matrix


def calibration_signal(q):
    n = q['source'][0][0]
    p = np.sin(q['before'][0][2])**2
    characteristic = 1 / (1 + n * p * (1 - np.exp(-1j * q['lambda'])))
    return (np.exp(-1j * q['phase']) * characteristic).real


class Calibrated:
    def fit(self, records):
        signals = np.array([calibration_signal(row['input']) for row in records])
        values = np.array([row['value'] for row in records])
        self.visibility = float(signals @ values / (signals @ signals))
        return self

    def predict(self, inputs):
        return self.visibility * np.array([self.reading(q) for q in inputs], dtype=float)


class IndependentRays(Calibrated):
    def source_pgf(self, source, argument):
        n, r = source[:, 0], source[:, 1]
        mean = n + (2*n+1)*np.sinh(r)**2
        return np.prod(1 / (1 + mean*(1-argument)))

    def reading(self, q):
        source = np.array(q['source'])
        size = len(source)
        prep, before, after = [abs(mesh(size, q[k]))**2 for k in ('prep', 'before', 'after')]
        bank = np.zeros(size)
        bank[q['bank']] = 1
        gate = q['gate']

        def fringe(angle, parity_phase):
            # A ray accumulates its transfer phase along one classical path.
            exit_phase = np.exp(1j*angle*bank) @ after
            if parity_phase:
                exit_phase[gate['rails']] *= -1
            single = ((exit_phase @ before) * np.exp(-1j*angle*bank)) @ prep
            return self.source_pgf(source, single)

        value = fringe(q['lambda'], False)
        if gate:
            s = gate['parity']
            value = (value + s*fringe(q['lambda'], True)) / (1 + s*fringe(0, True))
        return float((np.exp(-1j*q['phase'])*value).real)
