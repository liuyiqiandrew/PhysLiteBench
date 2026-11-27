import numpy as np
from scipy.linalg import schur
from functools import lru_cache


@lru_cache(maxsize=128)
def normal_modes(n, field, parity):
    h, hop, pairing, phi = field
    g = np.zeros((2*n, 2*n))
    xx = (-hop + pairing * np.cos(phi)) / 2
    yy = (-hop - pairing * np.cos(phi)) / 2
    xy = -pairing * np.sin(phi) / 2
    for j in range(n):
        a, b = 2*j, 2*j+1
        c, d = (a+2) % (2*n), (b+2) % (2*n)
        edge = -parity if j == n-1 else 1
        g[a,b] -= h/2
        g[b,c] -= edge * xx
        g[a,d] += edge * yy
        g[b,d] -= edge * xy
        g[a,c] += edge * xy
    # Put any zero modes last, so they enter in pairs even at gap closings.
    t, basis, _ = schur(g - g.T, output="real",
                        sort=lambda re, im: abs(im) > 1e-12)
    return np.diag(t, 1)[::2], basis.T


def pfaffian(a):
    # Pairwise elimination with a row/column pivot preserves the complex sign.
    product = 1.
    for i in range(0, len(a)-1, 2):
        j = i + 1 + np.argmax(abs(a[i, i+1:]))
        if a[i, j] == 0:
            return 0j
        if j != i+1:
            a[[i+1,j], :] = a[[j,i+1], :]
            a[:, [i+1,j]] = a[:, [j,i+1]]
            product = -product
        pivot = a[i,i+1]
        product *= pivot
        u = a[i, i+2:]
        v = a[i+1, i+2:] / pivot
        a[i+2:, i+2:] += np.outer(v, u) - np.outer(u, v)
    return product


def trace(q, parity, thermal=False):
    n = q['n']
    elapsed = sum(dt for dt, _ in q['pulses'])
    # In a trace, write the latest operation at the left.
    segments = [(q['prep'], -q['beta'] if thermal else -q['beta'] + 1j*elapsed, False)]
    if not thermal:
        tagged = q.get('tag')
        for j, (dt, field) in enumerate(q['pulses']):
            segments.append((field, -1j*dt, j == tagged))
    vectors, coeff = [], []
    for field, exponent, tagged in reversed(segments):
        energies, modes = normal_modes(n, tuple(field), -parity if tagged else parity)
        if tagged:
            # X_0=c_0. Conjugation reverses signs of all other Majoranas.
            # A global minus on the basis cancels in a quadratic product.
            modes = modes.copy()
            modes[:, 0] *= -1
        vec = modes.astype(complex)
        vec[::2] *= (1j*np.sinh(exponent*energies))[:, None]
        vectors.extend(vec)
        coeff.extend(np.cosh(exponent*energies))

    def contracted(project):
        v = np.vstack([*vectors, *np.eye(2*n)]) if project else np.array(vectors)
        a = np.triu(v @ v.T, 1)
        idx = 2*np.arange(len(coeff))
        a[idx, idx+1] += coeff
        return pfaffian(a-a.T)

    projected = (contracted(False) + parity * (-1j)**n * contracted(True)) / 2
    return -projected if not thermal and q.get('tag') is not None else projected


def fringe(q):
    z = sum(trace(q,p,True) for p in (1,-1))
    amp = sum(trace(q,p) for p in (1,-1))
    return float((np.exp(-1j*q['angle']) * amp/z).real)


class Model:
    def fit(self, records):
        x = np.array([fringe(r['input']) / r['sigma'] for r in records])
        y = np.array([r['value'] / r['sigma'] for r in records])
        self.visibility = float(np.clip((x @ y)/(x @ x), .65, .95))
        return self

    def predict(self, inputs):
        return self.visibility * np.array([fringe(q) for q in inputs])
