"""Independent reduced friction equations on a refined spatial mesh."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

TRUE_DIFFUSIVITY = .035


def predict(t, x, initial, diffusivity, refinement=4):
    t, x, initial = np.asarray(t), np.asarray(x), np.asarray(initial)
    fine = np.linspace(x[0], x[-1], refinement*(len(x)-1)+1)
    state = np.column_stack([np.interp(fine, x, initial[:, k]) for k in range(2)])
    n = len(fine)
    dx = fine[1]-fine[0]
    weights = np.full(n, dx)
    weights[[0, -1]] *= .5
    d12, d13, d23 = diffusivity*np.array([.2, 1., 5.])

    def derivative(time, flat):
        fraction = flat.reshape(n, 2)
        a, b = ((fraction[1:]+fraction[:-1])/2).T
        g1, g2 = (np.diff(fraction, axis=0)/dx).T
        a11 = b/d12+(1-b)/d13
        a12 = a*(1/d13-1/d12)
        a21 = b*(1/d23-1/d12)
        a22 = a/d12+(1-a)/d23
        determinant = a11*a22-a12*a21
        flux = np.zeros((n+1, 2))
        flux[1:-1, 0] = (-a22*g1+a12*g2)/determinant
        flux[1:-1, 1] = (a21*g1-a11*g2)/determinant
        return (-np.diff(flux, axis=0)/weights[:, None]).ravel()

    if t[-1] == 0:
        return initial[None].copy()
    pattern = diags([np.ones(2*n-abs(k)) for k in range(-3, 4)],
                    range(-3, 4), format='csr')
    result = solve_ivp(derivative, (0., t[-1]), state.ravel(), t_eval=t,
                       method='BDF', rtol=3e-9, atol=3e-11, jac_sparsity=pattern)
    assert result.success, result.message
    two = result.y.T.reshape(len(t), n, 2)[:, ::refinement]
    return np.concatenate([two, 1-two.sum(axis=2, keepdims=True)], axis=2)


def hidden_inputs():
    x = np.linspace(0, 1, 41)
    t = np.array([.2, .5, 1., 2., 3., 5., 8., 12.])
    c = np.cos(np.pi*x)
    c2 = np.cos(2*np.pi*x)
    profiles = [(.375+.35*c, np.full_like(x, .25)),
                (np.full_like(x, .25), .375+.35*c),
                (.3+.17*c2, .25+.2*c)]
    return [(t, x, np.column_stack([a, b, 1-a-b])) for a, b in profiles]


def metrics(model, data):
    prediction = np.array([model.predict(data['t'], data['x'], p) for p in data['initial']])
    residual = (prediction-data['mole_fraction'])/data['sigma']
    result = {'parameter': float(model.diffusivity),
              'parameter_relative_error': abs(float(model.diffusivity)/TRUE_DIFFUSIVITY-1),
              'calibration_chi2': float(np.sum(residual**2)/(residual.size-1)), 'hidden': {}}
    for i, (t, x, initial) in enumerate(hidden_inputs()):
        truth = predict(t, x, initial, TRUE_DIFFUSIVITY)
        actual = model.predict(t, x, initial)
        assert actual.shape == truth.shape and np.isfinite(actual).all()
        equilibrium = np.trapezoid(initial, x, axis=0)/(x[-1]-x[0])
        scale = np.sqrt(np.mean((truth-equilibrium)**2))
        result['hidden'][f'mixture_{i}'] = float(np.sqrt(np.mean((actual-truth)**2))/scale)
    return result
