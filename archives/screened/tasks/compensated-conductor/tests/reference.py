"""Solve two steady Lorentz-force balances and the total transverse-current constraint."""
import numpy as np

PARAMETER = 'mobility'
TRUE_PARAMETER = .48
BOUNDS = (.08, 1.4)
CHARGE = 1.602176634e-19


def state(experiment, mobility):
    e = experiment
    b = mobility*e['magnetic_field']
    total = e['density_positive']+e['density_negative']
    fp, fm = e['density_positive']/total, e['density_negative']/total
    equations = np.array([[1., -b, 0., 0., 0.],
                          [b, 1., 0., 0., -mobility],
                          [0., 0., 1., b, 0.],
                          [0., 0., -b, 1., mobility],
                          [0., fp, 0., -fm, 0.]])
    rhs = [mobility*e['electric_field'], 0., -mobility*e['electric_field'], 0., 0.]
    return np.linalg.solve(equations, rhs)


def predict(experiments, mobility):
    result = []
    for e in experiments:
        vpx, vpy, vmx, vmy, ey = state(e, mobility)
        result.append(CHARGE*(e['density_positive']*vpx-e['density_negative']*vmx))
    return np.array(result)


def calibration_inputs():
    return [dict(electric_field=ex, magnetic_field=b, density_positive=n, density_negative=n)
            for n in [.6e21, 1.1e21, 1.5e21]
            for b in [-5., -3.5, -2., -1., 0., 1., 2., 3.5, 5.]
            for ex in [-2.4, -1.1, .8, 2.1]]


def hidden_inputs():
    result = {}
    for name, positive, negative in [('positive_majority', 2.7e21, .3e21),
                                      ('negative_majority', .3e21, 2.7e21),
                                      ('partial_compensation', 1.8e21, .8e21)]:
        result[name] = [dict(electric_field=ex, magnetic_field=float(b),
                            density_positive=positive, density_negative=negative)
                        for b in np.r_[np.linspace(-5.8, -1.8, 12), np.linspace(1.8, 5.8, 12)]
                        for ex in [-2.1, .8, 1.7]]
    return result
