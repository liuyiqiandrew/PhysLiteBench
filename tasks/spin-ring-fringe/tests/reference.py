"""Two spin-space references, independent of the fermionic solution."""
from itertools import product

import numpy as np

TRUE_PARAMETER = .817


def experiment(n, beta, prep, pulses, angle):
    return dict(n=n, beta=beta, prep=prep, pulses=pulses, angle=angle)


def hidden_inputs():
    quench, winding, echoes = [], [], []
    for n in [5, 6, 7, 8]:
        prep = [.35, .89, -.74, .49]
        pulses = [[.83, [-.77, .55, .93, .81]], [1.32, [.45, -.33, .66, -1.11]]]
        for angle in [0, np.pi / 2]:
            quench.append(experiment(n, 3.45, prep, pulses, angle))
    for n in [9, 12, 17, 22, 27, 28]:
        pulses = [[1.58, [0, 1., -1., .7]]]
        winding.append(experiment(n, 0, [0, 0, 0, 0], pulses, n * np.pi / 2))
    for n in [11, 14, 19, 24]:
        prep = [.96, 0, 0, 0]
        pulses = [[1.53, [0, 1, -1, .8]], [.27, [.62, 0, 0, 0]],
                  [.46, [0, -.67, .67, 2.1]]]
        for angle in [0, np.pi/2]:
            echoes.append(experiment(n, 2.8, prep, pulses, angle))
    for family in (quench, winding, echoes):
        for q in family:
            q["tag"] = 0
    return {"thermal-quench": quench, "winding": winding, "pulse-echo": echoes}


def spin_hamiltonian(n, setting):
    h, hop, pair, phase = setting
    states = np.arange(2**n)
    matrix = np.diag([complex(h * (int(s).bit_count() - n / 2)) for s in states])
    for j in range(n):
        k = (j + 1) % n
        bit_j, bit_k = (states >> j) & 1, (states >> k) & 1
        coefficient = np.where(bit_j != bit_k, -hop, pair * np.exp(1j * phase * (1 - 2 * bit_j)))
        matrix[states ^ ((1 << j) | (1 << k)), states] += coefficient
    return matrix


def spin_trace(expt):
    n = expt["n"]
    total = sum(dt for dt, _ in expt["pulses"])
    e, vec = np.linalg.eigh(spin_hamiltonian(n, expt["prep"]))
    thermal = np.exp(-expt["beta"] * (e - e.min()))
    matrix = (vec * (thermal * np.exp(1j * total * e))) @ vec.conj().T
    for j, (dt, setting) in enumerate(expt["pulses"]):
        e, vec = np.linalg.eigh(spin_hamiltonian(n, setting))
        if expt.get("tag") == j:
            vec = vec[np.arange(2**n) ^ 1]
            matrix = -matrix
        matrix = (vec * np.exp(-1j * dt * e)) @ (vec.conj().T @ matrix)
    return np.trace(matrix) / thermal.sum()


def spatial_trace(expt):
    """Contract histories in the eigenbasis of each commuting spin layer."""
    n, beta = expt["n"], expt["beta"]
    h, *bonds = expt["prep"]
    assert not any(bonds)
    total = sum(dt for dt, _ in expt["pulses"])
    signs = np.array([1., -1.])
    bases, fields, edges = [np.eye(2)], [(1j * total - beta) * h / 2], [0.]
    for dt, (field, hop, pair, phase) in expt["pulses"]:
        if hop == pair == 0:
            basis, onsite, edge = np.eye(2), -1j * dt * field / 2, 0.
        else:
            assert field == 0 and pair == -hop
            basis = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            basis = basis.astype(complex)
            basis[1] *= np.exp(-.5j * phase)
            onsite, edge = 0., 1j * dt * hop
        bases.append(basis)
        fields.append(onsite)
        edges.append(edge)
    # A column is one site's list of spin outcomes, one per time layer.
    columns = np.array(list(product(range(2), repeat=len(bases))))
    spins = signs[columns]
    bonds = np.exp((spins * edges) @ spins.T)
    normalization = 2 * np.cosh(beta * h / 2)

    def transfer(local_bases):
        local = np.exp(spins @ fields)
        for i, basis in enumerate(local_bases):
            overlap = basis.conj().T @ local_bases[i - 1]
            local *= overlap[columns[:, i], columns[:, i - 1]]
        return local[:, None] * bonds / normalization

    ordinary = transfer(bases)
    tag = expt.get("tag")
    if tag is None:
        return np.trace(np.linalg.matrix_power(ordinary, n))
    site_zero = bases.copy()
    site_zero[tag+1] = site_zero[tag+1][::-1]  # the two physical pi rotations
    return -np.trace(transfer(site_zero) @ np.linalg.matrix_power(ordinary, n-1))


def predict(inputs, visibility):
    values = []
    for expt in inputs:
        z = spin_trace(expt) if expt["n"] <= 8 else spatial_trace(expt)
        values.append(visibility * (np.exp(-1j * expt["angle"]) * z).real)
    return np.array(values)
