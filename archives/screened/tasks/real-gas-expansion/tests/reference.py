import numpy as np

from scipy.integrate import quad

def predict(experiments, heat_capacity):
    result = []
    for e in experiments:
        if e['protocol'] == 'heating':
            result.append(heat_capacity*(e['final_temperature']-e['initial_temperature']))
        else:
            work_on_internal = quad(lambda v: .36/v**2, e['initial_volume'], e['final_volume'])[0]
            result.append(e['initial_temperature']-work_on_internal/heat_capacity)
    return np.array(result)


def calibration_inputs():
    return [dict(protocol='heating', initial_temperature=t, final_temperature=t+float(d), volume=v)
        for t, v in [(310., .0004), (340., .0007), (370., .0012)] for d in np.linspace(-8., 60., 32)]

def hidden_inputs():
    return {f'expansion_{j}': [dict(protocol='expansion', initial_temperature=t,
               initial_volume=v, final_volume=float(w)) for w in np.linspace(v*1.3, v*3., 24)]
            for j, (t, v) in enumerate([(450., .00035), (480., .0005), (520., .0008)])}

PARAMETER = 'heat_capacity'
TRUE_PARAMETER = 20.8
BOUNDS = (15.0, 40.0)
