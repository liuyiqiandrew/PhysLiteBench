"""Magnetic heat kernel in a periodic position box, with finite screws at the material origin."""
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import expm_multiply
from reference import grid, rates


def reference_area(q, nu, order=6, points=27, box=11.8, h=0.002):
    s = q["area"]
    mu = s["field"]
    a, b = s["axes"]
    fa = abs(mu)
    factor = np.sqrt(fa)
    t1, t2 = q["times"]
    k1, k2 = q["wave"]
    end = np.array(s["end"])
    checkpoint = s["middle"]
    u, w, Y, adj, neigh = grid(order, h)
    n = Y.shape[1]
    g0, g1 = [(1 + u @ g) ** 2 / 4 for g in q["gates"]]
    tag = u @ q["tag"]
    left = (w * g0 * np.exp(-1j * k1 * tag)) @ Y
    mid = csr_matrix(adj @ ((g1 * np.exp(1j * (k1 - k2) * tag))[:, None] * Y))
    tail = adj @ np.exp(1j * k2 * tag)

    def orbital(k):
        G = -k * k * (u**2 @ [0.18, 0.27, 0.36])[:, None] * Y
        first = []
        for rate, pair in zip(rates, neigh):
            f, b = [np.exp(1j * k * q["hand"] * dz)[:, None] * mat for mat, dz in pair]
            G += rate * (f + b - 2 * Y) / h**2
            M = adj @ ((f - b) / (2 * h))
            M[abs(M) < 5e-8] = 0.0
            first.append(csr_matrix(M))
        G = adj @ G
        G[abs(G) < 5e-8] = 0.0
        return csr_matrix(G), first

    zero, ds = orbital(0)
    small = zero[:9, :9].toarray()
    first = [ds[i][:9, :9].toarray() for i in (a, b)]
    time = t1 + t2 if checkpoint is None else t1
    endpoint = end if checkpoint is None else np.asarray(checkpoint)
    l0 = ((w * g0) @ Y)[:9]
    r0 = (adj @ g1)[:9]
    xs = np.linspace(-6.6, 6.6, 27)
    step = xs[1] - xs[0]
    den = 0j
    scale = np.sqrt(nu * time)
    for i in xs:
        mat = small + 2j * np.sqrt(rates[a]) * i * first[0] / scale
        for j in xs:
            z = mat + 2j * np.sqrt(rates[b]) * j * first[1] / scale
            den += np.exp(-i * i - j * j - 1j * (i * endpoint[0] + j * endpoint[1]) / scale) * (
                l0 @ expm(nu * t1 * z) @ r0
            )
    den *= np.exp(endpoint @ endpoint / (4 * nu * time)) * step**2 / np.pi
    x = (np.arange(points) - (points - 1) / 2) * box / points
    momentum = 2 * np.pi * np.fft.fftfreq(points, d=box / points)
    F = np.exp(-1j * np.outer(momentum, x)) / np.sqrt(points)
    p = F.conj().T @ (momentum[:, None] * F)
    lap = F.conj().T @ ((momentum**2)[:, None] * F)
    energy = csr_matrix(-fa * (lap + np.diag(x * x)))
    ispace = eye(n, format="csr")
    iplane = eye(points, format="csr")

    def evolution(v, k, t):
        G, drift = orbital(k)
        M = kron(G, iplane, format="csr") + kron(ispace, energy, format="csr")
        M += 2j * np.sqrt(rates[a]) * factor * kron(drift[a], csr_matrix(p), format="csr")
        M += 2j * np.sqrt(rates[b]) * np.sign(mu) * factor * kron(drift[b], diags(x), format="csr")
        return expm_multiply(nu * t * M, v)

    def inflate(v):
        return np.kron(v[:, None], np.eye(points))

    def endpoint_coef(v, d, t):
        phase = np.exp(-0.5j * np.sign(mu) * factor * d[1] * x)
        shift = F.conj().T @ (np.exp(-1j * factor * d[0] * momentum)[:, None] * F)
        inverse = phase[:, None] * shift * phase[None, :]
        return (
            2
            * fa
            * nu
            * t
            * np.exp(d @ d / (4 * nu * t))
            * np.einsum("iaj,ja->i", v.reshape(n, points, points), inverse)
        )

    v = evolution(inflate(tail), k2, t2)
    if checkpoint is None:
        v = kron(mid, iplane, format="csr") @ v
        amp = endpoint_coef(evolution(v, k1, t1), end, t1 + t2)
    else:
        d1 = np.asarray(checkpoint)
        d2 = end - d1
        v = mid @ endpoint_coef(v, d2, t2)
        amp = endpoint_coef(evolution(inflate(v), k1, t1), d1, t1) * np.exp(
            0.5j * mu * (d1[0] * d2[1] - d1[1] * d2[0])
        )
    return float((np.exp(-1j * q["phase"]) * (left @ amp) / den).real)
