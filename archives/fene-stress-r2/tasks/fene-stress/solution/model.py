"""Stationary connector distribution in a general planar linear flow."""
import numpy as np
from scipy.special import roots_jacobi, eval_jacobi
from scipy.linalg import solve
from scipy.optimize import minimize_scalar


def spectral(extension, rotation, temperature, drag, length, degree=24):
    """Weighted disk-polynomial weak FP, no spring-force singular quadrature."""
    exponent = length**2/(2*temperature)
    nodes, weights = roots_jacobi(degree+4, exponent, 0)
    x = (nodes+1)/2
    r = np.sqrt(x)
    theta = 2*np.pi*np.arange(4*degree+8)/(4*degree+8)
    rr, tt = np.meshgrid(r, theta, indexing='ij')
    w = np.broadcast_to((weights/weights.sum())[:, None]/len(theta), rr.shape).ravel()
    vals, radial, angular = [], [], []
    for total in range(0, degree+1, 2):
        for m in range(0, total+1, 2):
            n = (total-m)//2
            p = eval_jacobi(n, exponent, m, nodes)
            rad = r**m*p
            der = m*r**(m-1)*p if m else np.zeros_like(r)
            if n:
                der += 2*r**(m+1)*(n+exponent+m+1)*eval_jacobi(n-1, exponent+1, m+1, nodes)
            for sine in ([False] if m==0 else [False, True]):
                angle = np.sin(m*theta) if sine else np.cos(m*theta)
                angle_der = m*np.cos(m*theta) if sine else -m*np.sin(m*theta)
                z = (rad[:, None]*angle).ravel()
                norm = np.sqrt(np.dot(w, z*z))
                vals.append(z/norm)
                radial.append((der[:, None]*angle).ravel()/norm)
                angular.append((rad[:, None]*angle_der/rr).ravel()/norm)
    v, gr, gt = np.array(vals), np.array(radial), np.array(angular)
    radius, angle = rr.ravel(), tt.ravel()
    diffusion = 2*temperature/(drag*length**2)
    advection = (extension*radius*np.cos(2*angle))*gr + (rotation-extension*np.sin(2*angle))*radius*gt
    a = -diffusion*((gr*w)@gr.T+(gt*w)@gt.T)+(advection*w)@v.T
    coefficients = np.r_[1., solve(a[1:, 1:], -a[1:, 0])]
    f = coefficients@v
    prob = w*f
    cx = length**2*np.dot(prob, radius**2*np.cos(angle)**2)
    cy = length**2*np.dot(prob, radius**2*np.sin(angle)**2)
    cxy = length**2*np.dot(prob, radius**2*np.cos(angle)*np.sin(angle))
    c = np.array([[cx,cxy],[cxy,cy]])
    stress = drag/2*(extension*np.trace(c)-2*rotation*cxy)
    force = length**2*np.dot(prob, radius**2*np.cos(2*angle)/(1-radius**2))
    return float(stress), c, {'minimum_relative_density':float(f.min()),
        'negative_probability_mass':float(-np.minimum(prob,0).sum()),
        'normalization_error':float(abs(prob.sum()-1)),
        'weak_residual':float(np.max(abs(a@coefficients))),
        'force_moment_error':float(abs(force-stress))}


def predict_at(experiments, drag):
    values = []
    for e in experiments:
        rate, rotation, temperature = e['rate'], e['rotation'], e['temperature']
        if e['max_length'] is None:
            a = 2/drag
            values.append(2*temperature*rate*a/(a*a+rotation*rotation-rate*rate))
        else:
            values.append(spectral(rate, rotation, temperature, drag, e['max_length'])[0])
    return np.array(values)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        inputs = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def objective(drag):
            residual = (predict_at(inputs, drag)-values)/sigma
            return residual@residual
        result = minimize_scalar(objective, bounds=(3.2,4.8), method='bounded',
                                 options={'xatol':1e-12})
        candidates = [result.x, 3.2, 4.8]
        self.drag = float(min(candidates, key=objective))
        return self

    def predict(self, experiments):
        if self.drag is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments, self.drag)
