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

    # This is the two-projective-measurement distribution.  For initial
    # outcome i and final outcome j, the record is E_j-E_i and the joint
    # probability is p_i |U[j, i]|^2.  Powers of U^dagger H U-H would not
    # give these moments because the two energy operators do not commute.
    transition_probabilities=np.abs(unitary)**2
    energy_changes=energies[:,None]-energies[None,:]
    joint_probabilities=transition_probabilities*populations[None,:]

    raw_moments=np.array([
        np.sum(joint_probabilities*energy_changes**power)
        for power in (1,2,3)
    ], dtype=float)
    mean=raw_moments[0]
    variance=raw_moments[1]-mean**2
    third=raw_moments[2]-3*mean*raw_moments[1]+2*mean**3
    return np.array([mean,variance,third], dtype=float)


def predict_at(experiments, scale):
    keys=['temperature','amplitude_a','amplitude_b','phase','time_a','time_b']
    cache={}
    values=[]
    for e in experiments:
        key=tuple(e[k] for k in keys)
        if key not in cache:
            cache[key]=statistics(*key,scale)
        values.append(cache[key][e['cumulant']-1])
    return np.array(values)


class Model:
    def __init__(self):
        self.energy_scale=None

    def fit(self, records):
        if not records:
            raise ValueError('at least one calibration record is required')

        inputs=[record['input'] for record in records]
        measured=np.asarray([record['value'] for record in records], dtype=float)
        sigma=np.asarray([record.get('sigma', 0.0003) for record in records], dtype=float)
        if (not np.isfinite(measured).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError('calibration values and uncertainties must be finite')

        def residual_sum_of_squares(scale):
            predictions=predict_at(inputs, float(scale))
            residual=(predictions-measured)/sigma
            return float(residual@residual)

        # The apparatus has one unknown parameter and specifies its physical
        # interval.  Fit every calibration record simultaneously, with its
        # stated Gaussian uncertainty as the weight.
        result=minimize_scalar(
            residual_sum_of_squares,
            bounds=(0.8, 1.2),
            method='bounded',
            options={'xatol': 1e-12},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('could not fit energy_scale')
        self.energy_scale=float(np.clip(result.x, 0.8, 1.2))
        return self

    def predict(self, experiments):
        if self.energy_scale is None or not np.isfinite(self.energy_scale):
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments,self.energy_scale)
