import numpy as np
from scipy.integrate import solve_ivp



class Model:
    def __init__(self):
        self.resistance = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        r = self.resistance
        out = []
        for e in experiments:
            t, i0, v = e['time'], e['initial_current'], e['voltage']
            l0 = .6+.4*e['x0']
            if e['amplitude'] == 0 or e['omega'] == 0:
                out.append(v/r+(i0-v/r)*np.exp(-r*t/l0))
            elif t == 0:
                out.append(i0)
            else:
                def rhs(s, current):
                    inductance = .6+.4*(e['x0']+e['amplitude']*np.sin(e['omega']*s))
                    return (v-r*current)/inductance
                sol = solve_ivp(rhs, [0, t], [i0], rtol=1e-9, atol=1e-11)
                out.append(sol.y[0, -1])
        return np.array(out)
