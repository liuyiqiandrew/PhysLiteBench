from math import erf

import numpy as np
from scipy.linalg import eigh_tridiagonal

# Midpoints in the radial eccentric anomaly on the outgoing half-orbit.
COSINE = np.cos((np.arange(128) + 0.5) * np.pi / 128)


def phi(r, field):
    m, b = field
    return -m / (b + np.hypot(b, r))


def orbit_bin(q):
    m, b = q["steps"][-1] if q["steps"] else q["slow"]
    rp, ra = q["peri"], q["apo"]
    sp, sa = np.hypot(b, [rp, ra])
    energy = -m / (sp + sa)
    angular = rp * ra * np.sqrt(-2 * energy / ((sp + b) * (sa + b)))
    period = 2 * np.pi * m / (-2 * energy)**1.5
    return energy, angular, 8 * np.pi**2 * angular * period


def phase_density(energy, angular, fields, exponent):
    m, b = fields[-1]
    # Unbound orbits have escaped. Use a finite placeholder for quadrature.
    bound = energy < 0
    e = np.where(bound, energy, -0.1)
    if len(fields) == 1:
        jr = m / np.sqrt(-2 * e) - (angular + np.hypot(angular, 2 * np.sqrt(m*b))) / 2
        initial = -0.5 / (jr + (angular + np.hypot(angular, 2)) / 2)**2
        return np.where(bound, (-initial / 0.4)**exponent, 0)

    center = -m / (2 * e)
    amplitude = np.sqrt(np.maximum((center - b)**2 + angular**2 / (2 * e), 0))
    s = center[..., None] + amplitude[..., None] * COSINE
    r = np.sqrt(np.maximum(s*s - b*b, 0))
    previous_e = e[..., None] + phi(r, fields[-2]) - phi(r, fields[-1])
    previous = phase_density(previous_e, angular, fields[:-1], exponent)
    return np.where(bound, np.mean(previous * s / center[..., None], axis=-1), 0)


def offset(L, m, b):
    return (L + np.hypot(L, 2 * np.sqrt(m*b))) / 2


def potential(r, m, b):
    return -m / (b + np.hypot(b, r))


def variance(J, L, field, strength, order=72):
    m, b = field
    beta_m, beta_b = strength
    momentum = J + offset(L, m, b)
    axis = momentum**2 / m
    omega = m**2 / momentum**3
    span = np.sqrt(np.maximum((axis - b)**2 - L**2 * axis/m, 0))
    theta = np.pi * (np.arange(order) + .5) / order
    s = axis[:, None] + span[:, None] * np.cos(theta)
    weight = s / axis[:, None]
    field_deriv = -m / (b+s) * (beta_m - beta_b*b/s)
    centered = field_deriv - np.mean(weight * field_deriv, axis=1)[:, None]
    return np.mean(weight * centered**2, axis=1) / omega**2


def grid_faces(cap, sigma, bins):
    # Resolve a narrow absorbing layer on its own length scale.
    if sigma < 8 * cap / bins:
        split = cap - 8 * sigma
        return np.r_[np.linspace(0, split, bins + 1)[:-1],
                     np.linspace(split, cap, 81)]
    return np.linspace(0, cap, bins + 1)


def solve(L, field, strength, tau, cap, exponent, sigma, bins=280):
    faces = grid_faces(cap, sigma, bins)
    widths = np.diff(faces)
    x = (faces[1:] + faces[:-1]) / 2
    f = (0.5 / 0.4 / (x + offset(L, 1, 1))**2)**exponent
    distances = np.r_[widths[0] / 2, np.diff(x), widths[-1] / 2]
    conductance = variance(faces, L, field, strength) / (2 * distances)
    conductance[0] = 0.
    diag = -(conductance[:-1] + conductance[1:]) / widths
    offdiag = conductance[1:-1] / np.sqrt(widths[:-1] * widths[1:])
    vals, vecs = eigh_tridiagonal(diag, offdiag)
    roots = np.sqrt(widths)
    density = (vecs @ (np.exp(tau * vals) * (vecs.T @ (f * roots)))) / roots
    return np.r_[0., x, cap], np.r_[1.5*density[0] - 0.5*density[1], density, 0.]


def eccentric_anomaly(theta, ecc):
    theta = (theta + np.pi) % (2*np.pi) - np.pi
    psi = theta / (1 + ecc)
    for _ in range(13):
        psi -= (psi + ecc*np.sin(psi) - theta) / (1 + ecc*np.cos(psi))
    return psi


def amplitude(axis, m, b, L):
    return np.sqrt(np.maximum((axis-b)**2 - L**2*axis/m, 0.))


def coherent_variance(J, L, field, strength, hold, order=192):
    m, b = field
    p = J + offset(L, m, b)
    axis = p[:,None]**2/m
    ecc = amplitude(axis,m,b,L) / axis
    omega = m**2 / p[:,None]**3
    phi = 2*np.pi*(np.arange(order)+.5)/order
    s = axis * (1 + ecc*np.cos(phi))
    after = eccentric_anomaly(phi + ecc*np.sin(phi) + omega*hold, ecc)
    end = axis * (1 + ecc*np.cos(after))
    def V(s):
        return -m/(b+s) * (strength[0] - strength[1]*b/s)
    return np.mean((V(s) - V(end))**2 * s/axis, axis=1) / omega[:,0]**2


def chamber_density(action,L,field,train,exponent,bins=220):
    m,b=field
    hold=train["hold"]
    C=offset(L,m,b)
    cap=m/np.sqrt(2*train["binding_limit"])-C
    def initial(j):
        return (.5/.4/(j+offset(L,1,1))**2)**exponent
    cycles=m*m*hold/(2*np.pi*(action+C)**3)
    high_harmonic = np.ceil(cycles)
    low_harmonic = np.floor(cycles)
    if abs(cycles - round(cycles)) < 1e-13 and round(cycles):
        return initial(action)
    def resonant(n):
        return np.cbrt(m*m*hold/(2*np.pi*n)) - C if n else np.inf
    lo=max(0.,resonant(high_harmonic))
    hi=min(cap,resonant(low_harmonic))
    absorbing = cap <= resonant(low_harmonic)
    span=hi-lo
    v=lambda j:coherent_variance(j,L,field,train['coupling'],hold)
    sigma=np.sqrt(train['exposure']*v(np.array([hi]))[0]) if absorbing else 0
    if not any(train["coupling"]):
        return initial(action)
    if absorbing and 0 < sigma < 1e-7*span and cap-action<10*sigma:
        return initial(action)*erf((cap-action)/(np.sqrt(2)*sigma))
    faces=np.linspace(lo,hi,bins+1)
    if absorbing and 1e-7*span<=sigma<8*span/bins:
        split=hi-8*sigma
        faces=np.r_[np.linspace(lo,split,bins+1)[:-1],np.linspace(split,hi,81)]
    dx=np.diff(faces)
    x=(faces[1:]+faces[:-1])/2
    d=v(faces)/(2*np.r_[dx[0]/2,np.diff(x),dx[-1]/2])
    d[0]=0
    if not absorbing or sigma<1e-7*span:
        d[-1]=0.
    diagonal=-(d[:-1]+d[1:])/dx
    offdiag=d[1:-1]/np.sqrt(dx[:-1]*dx[1:])
    vals,vec=eigh_tridiagonal(diagonal,offdiag)
    f=(vec @ (np.exp(train['exposure']*vals)*(vec.T @ (initial(x)*np.sqrt(dx)))))/np.sqrt(dx)
    atlo=initial(lo) if lo>0 else 1.5*f[0]-.5*f[1]
    athi=0. if absorbing and sigma>=1e-7*span else initial(hi)
    return np.interp(action,np.r_[lo,x,hi],np.r_[atlo,f,athi])


def train_density(e, L, field, train, exponent):
    m, b = field
    cut = train["binding_limit"]
    if e >= -cut:
        return 0.
    cap = m / np.sqrt(2*cut) - offset(L, m, b)
    action = m / np.sqrt(-2*e) - offset(L, m, b)
    if "hold" in train:
        return chamber_density(action, L, field, train, exponent)
    strength, tau = train["coupling"], train["exposure"]
    initial = (0.5 / 0.4 / (action + offset(L, 1, 1))**2)**exponent
    sigma = np.sqrt(tau * variance(np.array([cap]), L, field, strength)[0])
    if sigma == 0:
        return initial
    if sigma < 1e-7 * cap:
        # Locally constant diffusivity and flat initial density to O(sigma).
        # The half-line absorbing solution is erf(distance / sqrt(2 B tau)).
        distance = m * (-e - cut) / (
            np.sqrt(2 * cut * -e) * (np.sqrt(-e) + np.sqrt(cut)))
        return initial * erf(distance / (np.sqrt(2) * sigma))
    x, f = solve(L, field, strength, tau, cap, exponent, sigma)
    return np.interp(action, x, f)


class Model:
    def fit(self, records):
        x, y = [], []
        for record in records:
            e, _, volume = orbit_bin(record["input"])
            x.append(np.log(-e / 0.4))
            y.append(np.log(record["value"] / volume))
        self.exponent = float(np.dot(x, y) / np.dot(x, x))
        return self

    def predict(self, experiments):
        values = []
        for q in experiments:
            e, angular, volume = orbit_bin(q)
            fields = [q["slow"], *q["steps"]]
            if "train" in q:
                density = train_density(e, angular, q["slow"], q["train"], self.exponent)
            else:
                density = phase_density(e, angular, fields, self.exponent)
            values.append(volume * density)
        return np.array(values)
