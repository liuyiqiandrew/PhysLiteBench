from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


I2=np.eye(2)
X=np.array([[0,1],[1,0]],complex)
Z=np.diag([1.,-1.])


def frame(phi,theta,q,r):
    g=np.kron(I2,X)
    j=np.kron(np.cos(theta)*Z+np.sin(theta)*X,Z)
    first=np.cos(q*phi/2)*np.eye(4)-1j*np.sin(q*phi/2)*g
    second=np.cos(r*phi/2)*np.eye(4)-1j*np.sin(r*phi/2)*j
    u=first@second
    derivative=(-1j*q/2*g)@u+first@(-1j*r/2*j)@second
    return u,derivative


@lru_cache(maxsize=512)
def levels(theta,q,r):
    samples=[]
    for phi in 2*np.pi*np.arange(128)/128:
        u,derivative=frame(phi,theta,q,r)
        low=u[:,:2]
        dlow=derivative[:,:2]
        samples.append(dlow.conj().T@(np.eye(4)-low@low.conj().T)@dlow)
    coefficients=np.fft.fft(samples,axis=0)/len(samples)
    n=np.arange(-18,19)
    shift=((q+r)%2)/2
    matrix=np.diag(np.repeat((n-shift)**2,2)).astype(complex)
    for i,ni in enumerate(n):
        for j,nj in enumerate(n):
            matrix[2*i:2*i+2,2*j:2*j+2]+=coefficients[(ni-nj)%len(samples)]
    return eigh(matrix/2,eigvals_only=True)


def predict_at(experiments,inertia):
    """Return the canonical mean kinetic energy for each experiment.

    ``levels`` contains the dimensionless eigenvalues of the limiting
    kinetic operator.  The physical energies are those values divided by
    the moment of inertia.  Subtracting the lowest exponent before taking
    the Boltzmann weights avoids underflow at the cold end of the stated
    temperature range.
    """
    result = []
    for experiment in experiments:
        temperature = float(experiment['temperature'])
        energy = levels(
            float(experiment['theta']),
            int(experiment['q']),
            int(experiment['r']),
        ) / inertia
        weight = np.exp(-(energy - energy.min()) / temperature)
        result.append(float(np.dot(weight, energy) / weight.sum()))
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.inertia=None

    def fit(self,records):
        """Fit the single unknown moment of inertia by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(observed).all() or
                not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            raise ValueError('calibration values and uncertainties must be finite')

        def objective(inertia):
            residual = (predict_at(experiments, inertia) - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.8, 1.2),
            method='bounded',
            options={'xatol': 1e-12},
        )
        # The bounded search is over the documented physical range.  The
        # explicit clipping also protects the public attribute from a tiny
        # optimizer round-off excursion at an endpoint.
        self.inertia = float(np.clip(result.x, 0.8, 1.2))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.inertia)
