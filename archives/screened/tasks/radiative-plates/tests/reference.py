import numpy as np

def predict(experiments, emissivity):
    result = []
    for e in experiments:
        eps2 = e['emissivity_2']
        matrix = [[1, -(1-emissivity)], [-(1-eps2), 1]]
        outgoing = np.linalg.solve(matrix, [emissivity*5.670374419e-8*e['temperature_1']**4,
                                          eps2*5.670374419e-8*e['temperature_2']**4])
        result.append(outgoing[0]-outgoing[1])
    return np.array(result)


def calibration_inputs():
    return [dict(temperature_1=float(t), temperature_2=t2, emissivity_2=1.)
        for t2 in [280., 350., 460.] for t in np.linspace(300., 650., 32)]

def hidden_inputs():
    return {f'panel_{eps}': [dict(temperature_1=float(t), temperature_2=t2, emissivity_2=eps)
        for t in np.linspace(310., 680., 24)] for eps, t2 in [(.15, 280.), (.35, 400.), (.65, 520.)]}

PARAMETER = 'emissivity'
TRUE_PARAMETER = 0.37
BOUNDS = (0.05, 0.95)
