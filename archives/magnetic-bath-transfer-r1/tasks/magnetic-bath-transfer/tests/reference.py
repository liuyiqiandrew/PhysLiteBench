"""Independent bath-to-bath frequency transmission integral."""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad

TRUE_PARAMETER = 1.13


@lru_cache(maxsize=None)
def _power(mass, spring, coupling, field, tx, ty, friction):
    def transmission(frequency):
        diagonal = spring - mass * frequency**2 - 1j * friction * frequency
        determinant = diagonal**2 - coupling**2 - field**2 * frequency**2
        return frequency**2 * (coupling**2 + field**2 * frequency**2) / abs(determinant)**2
    integral = quad(transmission, 0., np.inf, epsabs=2e-11, epsrel=2e-11, limit=200)[0]
    return 2 * friction**2 * (tx - ty) * integral / np.pi


def predict(experiments, friction=TRUE_PARAMETER):
    values = []
    for e in experiments:
        value = _power(e["mass"], e["spring"], e["coupling"], e["field"],
                       *e["temperatures"], friction)
        values.append(value if e["bath"] == 0 else -value)
    return np.asarray(values, dtype=float)


def calibration_inputs():
    unique = []
    for i, (m,k,h) in enumerate([(.15,.9,.3),(.3,1.1,.45),(.5,1.4,.55),(.8,1.7,.4),
                                (.2,1.6,.5),(.6,1.,.5),(.7,1.3,.35),(.4,.8,.4),
                                (.15,1.8,.55),(.8,.9,.3),(.35,1.25,.5),(.65,1.55,.45)]):
        unique.append(dict(mass=m,spring=k,coupling=h,field=0.,
                           temperatures=[1.7,.65] if i % 2 == 0 else [.7,1.85],bath=i%2))
    return [dict(e) for _ in range(24) for e in unique]


def hidden_inputs():
    groups = {"moderate_fields": [], "inertial_exchange": [], "bath_reversal": []}
    for i in range(12):
        groups["moderate_fields"].append(dict(mass=.35+.035*(i%6),spring=1.05+.055*i,
            coupling=.32+.017*i,field=(-1)**i*(.35+.025*i),temperatures=[1.65,.7],bath=i%2))
        groups["inertial_exchange"].append(dict(mass=.16+.045*i,spring=1.5-.04*i,
            coupling=.2+.022*i,field=(-1)**i*(.65+.035*i),temperatures=[1.8,.6],bath=i%2))
        groups["bath_reversal"].append(dict(mass=.28+.035*i,spring=.9+.06*i,
            coupling=.51-.016*i,field=(-1)**i*(.45+.055*i),temperatures=[.65,1.75],bath=i%2))
    return groups
