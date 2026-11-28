from functools import lru_cache
import numpy as np


@lru_cache(None)
def ring(N, aspect, t):
    n = np.arange(N)
    k = 2 * np.pi * (n + int(t == -1) / 2) / N
    e = 2 * np.sin(k / 2)
    b = N * aspect / 2
    symbol = 1j * np.exp(0.5j * k) * np.tanh(b * e)
    v = np.fft.fft(symbol) / N
    if t == -1:
        v *= np.exp(-1j * np.pi * n / N)
    idx = n[None, :] - n[:, None]
    C = v.real[idx % N] * np.where(idx < 0, t, 1)
    P = 0.0 if t == 1 else np.prod(np.tanh(b * e))
    log = np.log1p(np.exp(-2 * b * e)).sum()
    if t == 1:
        log -= N * aspect * np.tan(np.pi / (4 * N))
    return C, P, np.exp(log)


def channel(q, gain, N):
    x = np.rint(np.array(q["sites"]) * N).astype(int)
    S = np.concatenate([np.arange(a, b) for a, b in zip(x[::2], x[1::2])])
    R = S + 1
    n = np.arange(N)
    L = np.setdiff1d(n, S)
    V = np.setdiff1d(n, R)
    den = num = mref = zref = signed = flux = 0.0
    for twist in (1, -1):
        for p in (-1, 1):
            C, P, Z = ring(N, q["aspect"], -twist * p)
            op = np.linalg.det(C[np.ix_(S, R)])
            pop = (-1) ** int(N + len(S)) * np.linalg.det(C[np.ix_(L, V)])
            m = Z * (op + p * pop)
            norm = Z * (1 + p * P)
            if twist == 1:
                mref += m
                zref += norm
            if q["seam"] != "free" and q["seam"] != twist:
                continue
            if q["parity"] not in (0, p):
                continue
            w = np.exp(gain * q["bias"] * twist)
            den += w * norm
            num += w * m
            signed += w * twist * m
            flux += w * twist * norm
    score = num / den
    if q["read"] == "response":
        score = gain * (signed / den - flux / den * score)
    return float(score / (mref / zref))
