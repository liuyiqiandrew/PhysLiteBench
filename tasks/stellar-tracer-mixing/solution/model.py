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


def solve(L, field, strength, tau, cap, exponent, bins=280):
    dx = cap / bins
    x = (np.arange(bins)+.5)*dx
    f = (.5/.4/(x + offset(L,1,1))**2)**exponent
    # Face conductance for a cell-centered finite-volume grid. The absorbing
    # boundary lies half a cell from the final cell center.
    conductance = variance(np.arange(bins+1)*dx,L,field,strength)/(2*dx**2)
    conductance[0] = 0
    conductance[-1] *= 2
    values, eigenvectors = eigh_tridiagonal(-(conductance[:-1]+conductance[1:]),conductance[1:-1])
    density = eigenvectors @ (np.exp(tau*values) * (eigenvectors.T @ f))
    return x,density



def train_density(e, L, field, train, exponent):
    m, b = field
    cut = train["binding_limit"]
    if e >= -cut:
        return 0.
    cap = m / np.sqrt(2*cut) - offset(L, m, b)
    action = m / np.sqrt(-2*e) - offset(L, m, b)
    if not any(train["coupling"]):
        return (.5/.4/(action + offset(L, 1, 1))**2)**exponent
    x, f = solve(L, field, train["coupling"], train["exposure"], cap, exponent)
    # The density is regular at the circular endpoint and zero at the aperture.
    return np.interp(action, np.r_[0., x, cap], np.r_[1.5*f[0]-.5*f[1], f, 0.])


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
