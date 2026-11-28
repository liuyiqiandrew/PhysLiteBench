"""Fourier inversion of the compound Poisson spectra in the two reporter fields."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre


X = np.linspace(-24., 24., 3072, endpoint=False)
K = 2 * np.pi * np.fft.fftfreq(len(X), X[1] - X[0])
NODES, WEIGHTS = roots_legendre(220)


def components(count, sigma, rate, kicks, coupling, reset=False):
    """The two columns are ordinary and Ramsey-weighted densities."""
    paths = [(1, 0., 1.)]
    last = 0.
    for t, a, b in kicks:
        same = (1 + np.exp(-2 * rate * (t - last))) / 2
        paths = [(old * step, h + old * step * (a * coupling[0] + b * coupling[1]), p * chance)
                 for old, h, p in paths for step, chance in [(1, same), (-1, 1 - same)]]
        last = t
    marked = sum(p * np.exp(-.5 * sigma**2 * (K - h)**2) for _, h, p in paths)
    initial = np.exp(-.5 * (sigma * K)**2)
    atom = np.exp(-count)
    transform = np.exp(count * (np.array([initial, marked]) - 1)) - atom
    density = np.fft.fftshift(np.fft.fft(transform, axis=1), axes=1).T / 48.
    if reset:
        # A new independent defect realization at readout keeps the burn statistics intact.
        gain = np.exp(count * (marked[0] - 1))
        density[:, 1] = density[:, 0] * gain
        atoms = np.array([atom, atom * gain])
    else:
        atoms = np.array([atom, atom])
    return atoms, CubicSpline(X, density, extrapolate=False)


def reading(q, reset=False):
    if q.get("blank"):
        return np.cos(q["phase"])
    y, mag = q["gate"]
    candidates = {(y, mag)}
    if q["unsigned_b"]:
        candidates.add((y, -mag))
    u, v = q["mix"]
    det = 1 - u * v
    total = np.zeros((3, 2), complex)
    for domain in q["domains"]:
        ratio = domain["ratio"]
        couplings = [(1., 0.), (0., 1.), (1., ratio)]
        cols = [components(*pop, q["kicks"], c, reset) for pop, c in zip(domain["pockets"], couplings)]
        (ax, fx), (ay, fy), (az, fz) = cols
        w = domain["weight"]
        for m1, m2 in candidates:
            x, y = (m1-u*m2)/det, (m2-v*m1)/det
            if abs(x) < 1e-11 and abs(y) < 1e-11:
                total[0] += w*ax*ay*az
                continue
            on_line = False
            for j, (coord, residual) in enumerate([(x, y), (y, x), (x, y-ratio*x)]):
                if abs(residual) < 1e-10:
                    direction = couplings[j]
                    scale = max(abs(direction[0]+u*direction[1]), abs(v*direction[0]+direction[1]))
                    total[1] += w*cols[j][1](coord)*cols[(j+1)%3][0]*cols[(j+2)%3][0]/scale
                    on_line = True
            if on_line:
                continue
            both = az*fx(x)*fy(y) + ax*fz(x)*fy(y-ratio*x) + ay*fz(y/ratio)*fx(x-y/ratio)/abs(ratio)
            # Integrate the shared contribution over a range well beyond all retained densities.
            z = 11.*NODES
            both += 11.*(WEIGHTS @ (fz(z)*fx(x-z)*fy(y-ratio*z)))
            total[2] += w*both/abs(det)
    for den, num in total:
        if den.real > 1e-13:
            return float((np.exp(-1j*q["phase"])*num).real/den.real)
    raise ValueError("zero burn probability")
