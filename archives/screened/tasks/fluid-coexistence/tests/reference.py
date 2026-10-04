"""Independent convex-envelope construction followed by common-tangent refinement."""
from functools import lru_cache
import numpy as np
from scipy.optimize import root

PARAMETER = 'attraction'
TRUE_PARAMETER = 3.
BOUNDS = (2.6, 3.4)


def free_energy(v, temperature, attraction):
    return -temperature*np.log(v-1)-attraction/v


def slope(v, temperature, attraction):
    return -temperature/(v-1)+attraction/v**2


@lru_cache(None)
def phase_contacts(temperature, attraction):
    grid = 1+np.geomspace(1e-4, 1e4, 30000)
    energy = free_energy(grid, temperature, attraction)
    hull = []
    for i in range(len(grid)):
        while len(hull) >= 2:
            j, k = hull[-2:]
            left = (energy[k]-energy[j])/(grid[k]-grid[j])
            right = (energy[i]-energy[k])/(grid[i]-grid[k])
            if left < right:
                break
            hull.pop()
        hull.append(i)
    gaps = np.diff(hull)
    index = int(np.argmax(gaps))
    assert gaps[index] > 1
    initial = grid[[hull[index], hull[index+1]]]
    def common_tangent(volumes):
        liquid, vapor = volumes
        chord = (free_energy(vapor, temperature, attraction)-free_energy(liquid, temperature, attraction))/(vapor-liquid)
        return [slope(liquid, temperature, attraction)-chord,
                slope(vapor, temperature, attraction)-chord]
    answer = root(common_tangent, initial, tol=1e-11)
    assert np.max(abs(np.array(common_tangent(answer.x)))) < 1e-10
    liquid, vapor = answer.x
    assert 1 < liquid < vapor
    p = -slope(liquid, temperature, attraction)
    return float(liquid), float(vapor), float(p)


def predict(experiments, attraction):
    out = []
    for e in experiments:
        v, t = e['volume'], e['temperature']
        p = -slope(v, t, attraction)
        if t < 8*attraction/27:
            liquid, vapor, coexistence_pressure = phase_contacts(t, attraction)
            if liquid <= v <= vapor:
                p = coexistence_pressure
        out.append(p)
    return np.array(out)


def calibration_inputs():
    return [dict(temperature=t, volume=float(v)) for t in [1.4, 1.6, 1.9, 2.2]
            for v in np.linspace(1.4, 8., 25)]


def hidden_inputs():
    return {f'isotherm_{t}': [dict(temperature=t, volume=float(v)) for v in np.linspace(low, high, 36)]
            for t, low, high in [(.64, 2., 12.), (.72, 2., 9.), (.80, 2.2, 6.)]}
