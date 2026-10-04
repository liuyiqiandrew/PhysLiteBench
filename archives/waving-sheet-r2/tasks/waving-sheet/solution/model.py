"""Velocity response above a driven sheet in a polymer solution."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad_vec


BETA = .25
RELAXATION = 1.

def complex_viscosity(nu, frequency, beta=BETA, relaxation=RELAXATION):
    return nu*(beta+(1-beta)/(1-1j*frequency*relaxation))

def derivatives(nu, k, frequency, y, beta=BETA, relaxation=RELAXATION):
    """Streamfunction and first three derivatives, stable at creeping flow."""
    viscosity = complex_viscosity(nu, frequency, beta, relaxation)
    s = np.sqrt(k*k-1j*frequency/viscosity)
    delta = -1j*frequency/(viscosity*(s+k))
    y = np.asarray(y)
    decay = np.exp(-k*y)
    small = np.abs(delta*y) < .25
    divided_exponential = np.empty_like(y, dtype=complex)
    divided_exponential[small] = decay[small]*np.expm1(-delta*y[small])/delta
    divided_exponential[~small] = (np.exp(-s*y[~small])-decay[~small])/delta
    values = []
    for n in range(4):
        divided_power = (-1)**n*sum(s**(n-1-j)*k**j for j in range(n))
        values.append(frequency/k*(decay*((-k)**n-k*divided_power)
                                   -k*(-s)**n*divided_exponential))
    return np.asarray(values)

def polymer_stress(psi,viscosity,k,frequency):
    u,v = psi[1],-1j*k*psi[0]
    g = np.array([[1j*k*u,psi[2]],[1j*k*v,-1j*k*psi[1]]])
    gy = np.array([[1j*k*psi[2],psi[3]],[k*k*psi[1],-1j*k*psi[2]]])
    c = RELAXATION/(1-1j*frequency*RELAXATION)*(g+g.T)
    cy = RELAXATION/(1-1j*frequency*RELAXATION)*(gy+gy.T)
    stress = (1-BETA)*viscosity*.5*np.real(
        g@c.conj()+c@g.conj().T-u*np.conj(1j*k*c)-v*cy.conj())
    return stress[0,1]


def coefficients(viscosity,wave_number,frequency):
    k,w = wave_number,frequency
    viscosity_star = viscosity*(.25+.75/(1-1j*w))
    s = np.sqrt(k*k-1j*w/viscosity_star)
    return np.array([k,s]),np.array([w*s/(k*(s-k)),-w/(s-k)])


def reading(e,viscosity):
    k,w = e['wave_number'],e['frequency']
    p,c = coefficients(viscosity,k,w)
    if e['observable']=='pumping':
        boundary = -.5*np.real(np.sum(p*p*c))
        u,v = -p*c,-1j*k*c
        flux = .5*np.real(np.sum(u[:,None]*v.conj()[None,:]/
                    (p[:,None]+p.conj()[None,:])))
        def integrand(x):
            psi = derivatives(viscosity,k,w,x/k)
            return polymer_stress(psi,viscosity,k,w)/k
        polymer = quad_vec(integrand,0.,np.inf,epsabs=1e-10,epsrel=1e-10)[0]
        return float(boundary+(flux-polymer)/viscosity)
    field = c*np.exp(-p*e['height'])
    u = np.sum(-p*field)
    v = np.sum(-1j*k*field)
    return float({'u_real':u.real,'u_imag':u.imag,
                  'v_real':v.real,'v_imag':v.imag}[e['observable']])


def predict_at(experiments,viscosity):
    cache = {}
    out = []
    for e in experiments:
        key = tuple(sorted(e.items()))
        if key not in cache:
            cache[key] = reading(e,viscosity)
        out.append(cache[key])
    return np.array(out,dtype=float)


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self,records):
        inputs = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def objective(viscosity):
            residual = (predict_at(inputs,viscosity)-values)/sigma
            return float(residual@residual)
        fit = minimize_scalar(objective,bounds=(.7,1.4),method='bounded',
                              options={'xatol':1e-12})
        self.viscosity = float(min([fit.x,.7,1.4],key=objective))
        return self

    def predict(self,experiments):
        if self.viscosity is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments,self.viscosity)
