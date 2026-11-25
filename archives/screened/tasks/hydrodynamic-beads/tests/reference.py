"""Mean dynamics from Stokes force balance, independently integrated in SI units."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp

PARAMETER = 'stiffness'
TRUE_PARAMETER = .3
BOUNDS = (.1, .8)


@lru_cache(None)
def trajectory(separation, factor1, factor2, shift1, shift2, stiffness):
    radius = .5e-6
    distance = separation*1e-6
    eta = 1e-3
    diagonal = 1/(6*np.pi*eta*radius)
    off_diagonal = 1/(4*np.pi*eta*distance)
    spring = stiffness*1e-6*np.array([factor1, factor2])
    center = 1e-6*np.array([shift1, shift2])
    def velocity(time, position):
        force = spring*(center-position)
        return np.array([diagonal*force[0]+off_diagonal*force[1],
                         off_diagonal*force[0]+diagonal*force[1]])
    answer = solve_ivp(velocity, (0., 1.), np.zeros(2), method='DOP853',
                       rtol=2e-12, atol=1e-18, dense_output=True)
    assert answer.success
    return answer


def predict(experiments, stiffness):
    out = []
    for e in experiments:
        if e['measurement'] == 'variance':
            # Equipartition in SI, then m^2 -> micrometer^2.
            out.append(1.380649e-23*300/(stiffness*e['factors'][e['bead']]*1e-6)*1e12)
        else:
            answer = trajectory(e['separation'], *e['factors'], *e['shift'], stiffness)
            out.append(answer.sol(e['time'])[e['bead']]*1e6)
    return np.array(out)


def calibration_inputs():
    return [dict(measurement='variance', bead=i%2, separation=float(5+i/10),
                 factors=[float(.6+i/100), float(1.6-i/100)]) for i in range(100)]


def hidden_inputs():
    return {name: [dict(measurement='mean', bead=bead, separation=distance,
                       factors=factors, shift=shift, time=float(t))
                   for t in np.geomspace(.003, .7, 36)]
            for name, bead, distance, factors, shift in [
                ('right_response', 1, 5., [1., 1.], [.12, 0.]),
                ('left_response', 0, 8., [.7, 1.4], [0., -.1]),
                ('unequal_traps', 1, 6., [1.6, .6], [-.14, 0.])
            ]}
