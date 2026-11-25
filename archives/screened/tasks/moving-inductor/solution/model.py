from scipy.optimize import minimize_scalar
import numpy as np
from scipy.integrate import solve_ivp



class Model:
    def __init__(self):
        self.resistance = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(value):
            self.resistance = value
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(0.2, 8.0), method='bounded', options={'xatol': 1e-11})
        self.resistance = float(answer.x)
        return self

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
                def inductance(s):
                    return .6+.4*(e['x0']+e['amplitude']*np.sin(e['omega']*s))
                sol = solve_ivp(lambda s, flux: v-r*flux/inductance(s), [0, t],
                                [l0*i0], rtol=1e-9, atol=1e-11)
                out.append(sol.y[0, -1]/inductance(t))
        return np.array(out)
