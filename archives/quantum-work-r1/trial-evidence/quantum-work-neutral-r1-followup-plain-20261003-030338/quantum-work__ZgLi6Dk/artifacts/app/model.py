import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from functools import lru_cache

D=np.diag([0.,1.,2.35])
X=np.array([[0.,1.,.25],[1.,0.,.8],[.25,.8,0.]])
Y=np.array([[0.,-1j,.45j],[1j,0.,-.7j],[-.45j,.7j,0.]])


@lru_cache(maxsize=4096)
def evolution(scale, amplitude_a, amplitude_b, phase, time_a, time_b):
    initial=scale*D
    first=initial+amplitude_a*X
    second=initial+amplitude_b*(np.cos(phase)*X+np.sin(phase)*Y)
    return expm(-1j*time_b*second)@expm(-1j*time_a*first)


def state_and_evolution(temperature, amplitude_a, amplitude_b, phase, time_a, time_b, scale):
    energies=scale*np.diag(D)
    populations=np.exp(-energies/temperature)
    populations/=populations.sum()
    unitary=evolution(scale,amplitude_a,amplitude_b,phase,time_a,time_b)
    return energies,populations,unitary


def statistics(temperature, amplitude_a, amplitude_b, phase, time_a, time_b, scale):
    energies,populations,unitary=state_and_evolution(
        temperature,amplitude_a,amplitude_b,phase,time_a,time_b,scale)
    hamiltonian=np.diag(energies)
    response=unitary.conj().T@hamiltonian@unitary-hamiltonian
    mean=float(np.real(populations@np.diag(response)))
    centered=response-mean*np.eye(3)
    variance=float(np.real(populations@np.diag(centered@centered)))
    third=float(np.real(populations@np.diag(centered@centered@centered)))
    return np.array([mean,variance,third])


def predict_at(experiments, scale):
    if scale is None or not np.isfinite(scale):
        raise RuntimeError('Model must be fitted before predict() is called')

    keys=['temperature','amplitude_a','amplitude_b','phase','time_a','time_b']
    cache={}
    values=[]
    for e in experiments:
        key=tuple(e[k] for k in keys)
        if key not in cache:
            cache[key]=statistics(*key,scale)
        cumulant=e['cumulant']
        if cumulant not in (1, 2, 3):
            raise ValueError('cumulant must be one of 1, 2, or 3')
        values.append(cache[key][int(cumulant)-1])
    result=np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise FloatingPointError('non-finite prediction')
    return result


class Model:
    def __init__(self):
        self.energy_scale=None

    def fit(self, records):
        """Fit the common energy scale to the calibration records.

        The calibration uncertainty is independent and Gaussian, so the
        maximum-likelihood estimate is the bounded weighted least-squares
        minimum.  There is only one unknown parameter; a coarse scan before
        the bounded local minimization makes the result insensitive to small
        oscillations in the driven evolution.
        """
        records=list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        inputs=[]
        observed=[]
        sigmas=[]
        for record in records:
            inputs.append(record['input'])
            observed.append(float(record['value']))
            sigma=float(record['sigma'])
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError('calibration sigma must be finite and positive')
            sigmas.append(sigma)

        observed=np.asarray(observed, dtype=float)
        sigmas=np.asarray(sigmas, dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError('calibration values must be finite')

        def objective(scale):
            residual=(predict_at(inputs, float(scale))-observed)/sigmas
            return float(residual @ residual)

        lower, upper=.8, 1.2
        grid=np.linspace(lower, upper, 41)
        grid_values=np.array([objective(scale) for scale in grid])
        best_index=int(np.argmin(grid_values))

        # Refine around the best scan point.  At an endpoint, retain the
        # endpoint as a valid bounded optimum.
        if 0 < best_index < len(grid)-1:
            result=minimize_scalar(
                objective,
                bounds=(grid[best_index-1], grid[best_index+1]),
                method='bounded',
                options={'xatol': 1e-12},
            )
            scale=float(result.x)
            if result.fun > grid_values[best_index]:
                scale=float(grid[best_index])
        else:
            scale=float(grid[best_index])

        self.energy_scale=float(np.clip(scale, lower, upper))
        return self

    def predict(self, experiments):
        return predict_at(experiments,self.energy_scale)
