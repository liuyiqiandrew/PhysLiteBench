from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

TRUE_PARAMETER = .7


@lru_cache(512)
def reservoir_current(friction, k1, k2, coupling, memory, t1, t2):
    def spectrum(omega):
        drag = friction / (1 - 1j * omega * memory)
        left = k1 + coupling - omega**2 - 1j * omega * drag
        right = k2 + coupling - omega**2 - 1j * omega * drag
        transfer = coupling / (left * right - coupling**2)
        return 4 * omega**2 * drag.real**2 * abs(transfer)**2
    transmission = quad(spectrum, 0, np.inf, epsabs=2e-11, epsrel=2e-10, limit=300)[0]
    return (t1 - t2) * transmission / (2 * np.pi)


def predict(experiments, friction):
    out = []
    for e in experiments:
        k1, k2 = e['springs']
        c, tau = e['coupling'], e['memory']
        t1, t2 = e['temperatures']
        if e['readout'] == 'heat_current':
            value = reservoir_current(float(friction), k1, k2, c, tau, t1, t2)
            out.append(value if e['bath'] == 0 else -value)
        else:
            eigenvalues, vectors = np.linalg.eigh(np.array([[k1+c, -c], [-c, k2+c]]))
            alpha = e['mode']
            temperature = np.dot(vectors[:, alpha]**2, [t1, t2])
            drift = np.array([[0., 1., 0.], [-eigenvalues[alpha], 0., 1.],
                              [0., -friction/tau, -1/tau]])
            # The total modal force is one equilibrium Drude reservoir at its
            # weighted temperature; its equilibrium p column is (0,T,0).
            out.append(temperature * expm(e['lag'] * drift)[1, 1])
    return np.array(out)


def hidden_inputs():
    groups = {}
    for name, springs, coupling, memory in [
        ('weak_mixing', [1., 1.2], .12, .3),
        ('finite_memory', [.8, 1.4], .4, .7),
        ('long_memory', [1., 1.2], .16, 1.8),
    ]:
        experiments = []
        for temperatures in ([2., .5], [.6, 1.8], [1.7, .9]):
            for factor in (.85, 1., 1.2):
                for bath in (0, 1):
                    experiments.append(dict(springs=springs, coupling=coupling*factor,
                        memory=memory, temperatures=temperatures,
                        readout='heat_current', bath=bath))
        groups[name] = experiments
    return groups
