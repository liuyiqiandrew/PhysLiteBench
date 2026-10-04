import numpy as np

def predict(experiments, polarizability):
    result = []
    for e in experiments:
        def energy(x):
            field = e['field_offset']+e['gradient']*x
            p = polarizability*field
            return p*p/(2*polarizability)-p*field
        if e['observable'] == 'dipole':
            result.append(polarizability*(e['field_offset']+e['gradient']*e['position']))
        else:
            h = 1e-4
            result.append(-(energy(e['position']+h)-energy(e['position']-h))/(2*h))
    return np.array(result)


def calibration_inputs():
    return [dict(observable='dipole', field_offset=float(f), gradient=0., position=p)
          for p in [-.4, .3] for f in np.linspace(-4.5, 4.5, 50)]

def hidden_inputs():
    return {f'gradient_{g}': [dict(observable='force', field_offset=f, gradient=g, position=float(x))
            for x in np.linspace(-.8, .8, 24)] for g, f in [(.8, 2.), (-1.5, 3.), (2.5, -3.)]}

PARAMETER = 'polarizability'
TRUE_PARAMETER = 1.7
BOUNDS = (0.2, 5.0)
