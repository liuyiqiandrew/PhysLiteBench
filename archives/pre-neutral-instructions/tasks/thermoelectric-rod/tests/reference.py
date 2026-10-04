"""Independent conservative heat flux plus electrical work on a refined mesh."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

TRUE_CONDUCTIVITY = 1.25
CAPACITY = 1.5e6
RHO = 2e-5


def predict(t, x, initial, current, boundary, conductivity, refinement=4):
    t, x, initial = np.asarray(t), np.asarray(x), np.asarray(initial)
    fine = np.linspace(x[0], x[-1], refinement*(len(x)-1)+1)
    dx = fine[1]-fine[0]
    state = np.interp(fine, x, initial)[1:-1]

    def derivative(time, interior):
        temp = np.r_[boundary[0], interior, boundary[1]]
        face = (temp[:-1]+temp[1:])/2
        gradient = np.diff(temp)/dx
        s_face = 2e-4+4e-6*(face-300.)
        heat_flux = current*s_face*face-conductivity*gradient
        field = RHO*current+(2e-4+4e-6*(interior-300.))*(temp[2:]-temp[:-2])/(2*dx)
        return (-np.diff(heat_flux)/dx+current*field)/CAPACITY

    if t[-1] == 0:
        temperature = initial[None].copy()
    else:
        pattern = diags([np.ones(len(state)-abs(k)) for k in [-1, 0, 1]], [-1, 0, 1])
        result = solve_ivp(derivative, (0., t[-1]), state, t_eval=t, method='BDF',
                           rtol=2e-10, atol=1e-8, jac_sparsity=pattern)
        assert result.success, result.message
        temperature = np.column_stack([np.full(len(t), boundary[0]), result.y.T,
                                        np.full(len(t), boundary[1])])[:, ::refinement]
    # Integrate the affine Seebeck function independently with its exact midpoint rule.
    voltage = RHO*current*(x[-1]-x[0])+np.sum(
        (2e-4+4e-6*((temperature[:, 1:]+temperature[:, :-1])/2-300.))*np.diff(temperature), axis=1)
    return {'temperature': temperature, 'voltage': voltage}


def hidden_inputs():
    x = np.linspace(0, .01, 51)
    t = np.array([5., 15., 30., 60., 100., 160.])
    result = []
    for current, boundary, amplitude in [(2e5, (290., 330.), 8.),
                                          (-2e5, (290., 330.), 8.),
                                          (1.5e5, (315., 285.), -6.)]:
        initial = np.linspace(*boundary, len(x))+amplitude*np.sin(np.pi*x/.01)
        result.append((t, x, initial, current, boundary))
    return result


def metrics(model, data):
    residuals = []
    for i, initial in enumerate(data['initial']):
        output = model.predict(data['t'], data['x'], initial, data['current'][i], data['boundary'][i])
        for field in ['temperature', 'voltage']:
            residuals.extend(((output[field]-data[field][i])/data['sigma_'+field][i]).ravel())
    result = {'parameter': float(model.conductivity),
              'parameter_relative_error': abs(float(model.conductivity)/TRUE_CONDUCTIVITY-1),
              'calibration_chi2': float(np.sum(np.square(residuals))/(len(residuals)-1)), 'hidden': {}}
    for i, args in enumerate(hidden_inputs()):
        truth = predict(*args, TRUE_CONDUCTIVITY)
        actual = model.predict(*args)
        t, x, initial, current, boundary = args
        baseline = np.linspace(*boundary, len(x))
        scale = np.sqrt(np.mean((truth['temperature']-baseline)**2))
        assert actual['temperature'].shape == truth['temperature'].shape
        assert np.isfinite(actual['temperature']).all() and np.isfinite(actual['voltage']).all()
        result['hidden'][f'current_{i}'] = float(np.sqrt(np.mean((actual['temperature']-truth['temperature'])**2))/scale)
        result.setdefault('voltage_absolute_error', []).append(float(np.max(abs(actual['voltage']-truth['voltage']))))
    return result
