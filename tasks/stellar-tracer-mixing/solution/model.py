import numpy as np

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
            values.append(volume * phase_density(e, angular, fields, self.exponent))
        return np.array(values)
