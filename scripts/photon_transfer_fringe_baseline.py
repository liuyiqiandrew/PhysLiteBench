"""Factorize the photon labels into their exact single-photon spectral density operators."""
from itertools import permutations
import numpy as np
from scipy.special import roots_hermitenorm
from gaussian import mesh


def reduced(q, nodes, weights):
    s = q["spectral"]
    states = s["preparations"]
    n = len(states[0]["mean"])
    indices = np.stack(np.meshgrid(*([nodes] * n), indexing="ij"), axis=-1)
    wt = np.ones(indices.shape[:-1])
    for j in range(n):
        shape = [1] * n
        shape[j] = len(nodes)
        wt *= weights.reshape(shape)
    packets = []
    for state in states:
        mu = np.array(state["mean"])
        cov = np.array(state["cov"])
        x = indices - mu
        e = -0.25 * np.einsum("...i,ij,...j->...", x, np.linalg.inv(cov), x)
        phase = 0.5 * np.einsum(
            "...i,ij,...j->...", indices, state["chirp"], indices
        ) - indices @ np.array(state["delay"])
        packets.append(
            np.exp(e + 1j * phase)
            * np.sqrt(wt)
            / (np.linalg.det(cov) * (2 * np.pi) ** n) ** 0.25
        )
    if len(packets) > 1:
        c = complex(*s["weight"])
        d = complex(*s["marker"])
        a, b = packets
        packets = [a + d * c * b, np.sqrt(max(0.0, 1 - abs(d) ** 2)) * c * b]
    densities = []
    for i in range(n):
        rho = np.zeros((len(nodes), len(nodes)), complex)
        for packet in packets:
            psi = np.moveaxis(packet, i, 0).reshape(len(nodes), -1)
            rho += psi.conj() @ psi.T
        densities.append(rho / np.trace(rho))
    return densities


def cycles(perm):
    parts = []
    seen = set()
    for i in range(len(perm)):
        if i in seen:
            continue
        cycle = []
        j = i
        while j not in seen:
            seen.add(j)
            cycle.append(j)
            j = perm[j]
        parts.append(cycle)
    return parts


def transfer(q, order=38):
    s = q["spectral"]
    n = len(s["preparations"][0]["mean"])
    nodes, weights = roots_hermitenorm(order)
    weights = 0.91 * weights * np.exp(0.5 * nodes**2)
    nodes = 0.91 * nodes
    rho = reduced(q, nodes, weights)
    B = mesh(n, q["before"])
    A = mesh(n, q["after"])
    which, delay, chirp = s["plate"]
    ph = np.ones((len(nodes), n), complex)
    ph[:, which] = np.exp(1j * (-delay * nodes + chirp * nodes**2 / 2))
    gate = q["gate"]
    terms = [(1.0, ph)]
    if gate:
        diag = np.ones(n)
        diag[gate["rails"]] = -1
        terms = [(0.5, ph), (0.5 * gate["parity"], ph * diag)]
    branches = [(c, (A[None, :, :] * v[:, None, :]) @ B) for c, v in terms]
    bank = np.isin(np.arange(n), q["bank"])
    perms = [(p, cycles(p)) for p in permutations(range(n))]

    def chi(theta):
        total = 0j
        change = np.exp(1j * theta * bank)
        for wa, left in branches:
            for wb, right in branches:
                z = (left.conj().transpose(0, 2, 1) * change) @ right
                for perm, pcs in perms:
                    val = wa * wb
                    for cycle in pcs:
                        first, *other = cycle
                        k = rho[first] * z[:, first, perm[first], None]
                        for j in other:
                            k = (rho[j] * z[:, j, perm[j], None]) @ k
                        val *= np.trace(k)
                    total += val
        return total

    mass = chi(0.0).real
    angle = q["lambda"]
    return float(
        (chi(angle) * np.exp(-1j * (q["phase"] + angle * sum(bank)))).real / mass
    )


from pump import PumpSurvey


class Model(PumpSurvey):
    def reading(self, q):
        return transfer(q) if "spectral" in q else super().reading(q)
