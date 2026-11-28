"""Double-click optics with a shared, number-dependent quantum pump."""
from itertools import product
import numpy as np
from pulse import PulseSurvey
from gaussian import click, mesh, source_pairs


def packet(components, dim):
    state = np.zeros(dim, complex)
    for m, re, im in components:
        state[m] = re + 1j * im
    return state / np.linalg.norm(state)


def histories(q, before, middle, discard=False):
    spec = q["pump"]
    nmax = max(row[0] for row in spec["state"] + (spec["detect"] or []))
    psi = packet(spec["state"], nmax + 1)
    detector = packet(spec["detect"], nmax + 1) if spec["detect"] is not None else None
    num = np.arange(nmax + 1)
    free = np.exp(-0.5j * spec["kerr"] * num * (num - 1))

    def lower(state):
        return np.r_[np.sqrt(num[1:]) * state[1:], 0j]

    results = []
    for order in range(2):
        phase = 1 if order == 0 else np.exp(1j * q["switch"]["angle"])
        terms = [
            click(q["switch"]["clicks"][idx], basis)
            for idx, basis in [(order, before), (1 - order, middle)]
        ]
        for adds in product((False, True), repeat=2):
            operators = [
                (0 * u, v) if add else (u, 0 * v) for (u, v), add in zip(terms, adds)
            ]
            if any(not np.any(abs(u) + abs(v) > 1e-14) for u, v in operators):
                continue
            phi = (lower(psi) if adds[0] else psi) * free
            # Optional which-number record immediately after the first click.
            if discard:
                alternatives = np.diag(phi)
                for j, v in enumerate(alternatives):
                    alternatives[j] = lower(v) if adds[1] else v
                outgoing = (
                    alternatives if detector is None else alternatives @ detector.conj()
                )
            else:
                outgoing = lower(phi) if adds[1] else phi
                if detector is not None:
                    outgoing = np.vdot(detector, outgoing)
            results.append((operators, phase * np.asarray(outgoing).ravel()))
    return results


def pump_signal(q, points=64, discard=False):
    n = len(q["source"])
    mean, sq, _ = np.array(q["source"]).T
    norm = np.prod(1 / ((1 + mean) * np.cosh(sq)))
    bank = np.isin(np.arange(n), q["bank"]).astype(float)
    gauge = np.exp(
        2j * np.pi * np.arange(points)[:, None] * np.r_[bank, np.zeros(n)] / points
    )
    state = gauge[:, :, None] * source_pairs(q) * gauge[:, None, :]
    adj = state.conj().transpose(0, 2, 1)
    before = mesh(n, q["before"])
    mid = mesh(n, q["switch"]["middle"]) @ before
    full = mesh(n, q["after"]) @ mid
    paths = histories(q, before, mid, discard)
    eye = np.eye(2 * n)

    def characteristic(angle):
        d = np.exp(1j * angle * bank)
        out = full.conj().T @ (d[:, None] * full)
        net = eye.astype(complex)
        net[:n, :n] = out * d.conj()
        right = net @ state @ net.T
        eig = np.linalg.eigvals(adj @ right)
        pref = norm * np.exp(-0.5 * np.log1p(-eig).sum(axis=-1))
        gram = np.linalg.inv(eye - right @ adj)
        value = 0j
        for left, pl in paths:
            bra = [(v.conj(), u.conj()) for u, v in left]
            for ket, pr in paths:
                factor = np.vdot(pl, pr)
                if abs(factor) < 1e-15:
                    continue
                ket = [(u @ out.conj().T, v @ out.T) for u, v in reversed(ket)]
                u, v = [
                    np.pad(np.array(x), ((0, 0), (0, n))) for x in zip(*(bra + ket))
                ]
                contraction = (u + v @ adj) @ gram @ (right @ u.T + v.T)
                wick = (
                    contraction[:, 0, 1] * contraction[:, 2, 3]
                    + contraction[:, 0, 2] * contraction[:, 1, 3]
                    + contraction[:, 0, 3] * contraction[:, 1, 2]
                )
                value += factor * np.mean(pref * wick)
        return value / 4

    rate = characteristic(0.0).real
    result = (np.exp(-1j * q["phase"]) * characteristic(q["lambda"])).real / rate
    return float(result), float(rate)


class PumpSurvey(PulseSurvey):
    monitor = False

    def reading(self, q):
        if "pump" in q:
            return pump_signal(q, discard=self.monitor)[0]
        return super().reading(q)
