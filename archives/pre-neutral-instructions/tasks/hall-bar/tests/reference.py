import numpy as np

def predict(experiments, mobility):
    out = []
    for e in experiments:
        b = mobility*e['magnetic_field']
        conductivity = 2e21*1.602176634e-19*mobility/(1+b*b)*np.array([[1., b], [-b, 1.]])
        ey = -conductivity[1, 0]*e['electric_field']/conductivity[1, 1]
        out.append((conductivity @ [e['electric_field'], ey])[0])
    return np.array(out)


def calibration_inputs():
    return [{'electric_field': float(e), 'magnetic_field': 0.} for e in np.linspace(-2.5, 2.5, 100)]

def hidden_inputs():
    return {f'field_{b}': [{'electric_field': float(e), 'magnetic_field': b}
            for e in np.linspace(.2, 2.8, 24)] for b in [2., -3.5, 5.]}

PARAMETER = 'mobility'
TRUE_PARAMETER = 0.45
BOUNDS = (0.05, 1.5)
